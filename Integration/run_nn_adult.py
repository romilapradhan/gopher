import tqdm
import time
import pandas as pd
import copy
import random
from load_dataset import load, generate_random_dataset
from classifier import NeuralNetwork, LogisticRegression, SVM
from utils import *
from metrics import *  # include fairness and corresponding derivatives
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
from torch.autograd import grad

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

dataset = 'adult'
X_train, X_test, y_train, y_test = load(dataset)

X_train_orig = copy.deepcopy(X_train)
X_test_orig = copy.deepcopy(X_test)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = NeuralNetwork(input_size=X_train.shape[-1])
num_params = len(convert_grad_to_ndarray(list(clf.parameters())))
loss_func = logistic_loss_torch


def ground_truth_influence(X_train, y_train, X_test, X_test_orig, y_test):
    clf.fit(X_train, y_train, verbose=True)
    y_pred = clf.predict_proba(X_test)
    spd_0 = computeFairness(y_pred, X_test_orig, y_test, 0)

    delta_spd = []
    for i in range(len(X_train)):
        X_removed = np.delete(X_train, i, 0)
        y_removed = y_train.drop(index=i, inplace=False)
        clf.fit(X_removed, y_removed)
        y_pred = clf.predict_proba(X_test)
        delta_spd_i = computeFairness(y_pred, X_test_orig, y_test, 0) - spd_0
        delta_spd.append(delta_spd_i)

    return delta_spd


def computeAccuracy(y_true, y_pred):
    return np.sum((y_pred > 0.5) == y_true) / len(y_pred)


def del_L_del_theta_i(model, x, y_true, retain_graph=False):
    loss = loss_func(model, x, y_true)
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(loss, w, create_graph=True, retain_graph=retain_graph)


def del_f_del_theta_i(model, x, retain_graph=False):
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(model(torch.FloatTensor(x)), w, retain_graph=retain_graph)


def hvp(y, w, v):
    """ Multiply the Hessians of y and w by v."""
    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(convert_grad_to_tensor(first_grads), v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def hessian_one_point(model, x, y):
    x, y = torch.FloatTensor(x), torch.FloatTensor([y])
    loss = loss_func(model, x, y)
    params = [p for p in model.parameters() if p.requires_grad]
    first_grads = convert_grad_to_tensor(grad(loss, params, retain_graph=True, create_graph=True))
    hv = np.zeros((len(first_grads), len(first_grads)))
    for i in range(len(first_grads)):
        hv[i, :] = convert_grad_to_ndarray(grad(first_grads[i], params, create_graph=True)).ravel()
    return hv


# Compute multiplication of inverse hessian matrix and vector v
def s_test(model, xs, ys, v, hinv=None, damp=0.01, scale=25.0, r=-1, batch_size=-1, recursive=False, verbose=False):
    """ Arguments:
        xs: list of data points
        ys: list of true labels corresponding to data points in xs
        damp: dampening factor
        scale: scaling factor
        r: number of iterations aka recursion depth
            should be enough so that the value stabilises.
        batch_size: number of instances in each batch in recursive approximation
        recursive: determine whether to recursively approximate hinv_v"""
    xs, ys = torch.FloatTensor(xs.copy()), torch.FloatTensor(ys.copy())
    n = len(xs)
    if recursive:
        hinv_v = copy.deepcopy(v)
        if verbose:
            print('Computing s_test...')
            tbar = tqdm.tqdm(total=r)
        if batch_size == -1:  # default
            batch_size = 10
        if r == -1:
            r = n // batch_size + 1
        sample = np.random.choice(range(n), r * batch_size, replace=True)
        for i in range(r):
            sample_idx = sample[i * batch_size:(i + 1) * batch_size]
            x, y = xs[sample_idx], ys[sample_idx]
            loss = loss_func(model, x, y)
            params = [p for p in model.parameters() if p.requires_grad]
            hv = convert_grad_to_ndarray(hvp(loss, params, torch.FloatTensor(hinv_v)))
            # Recursively caclulate h_estimate
            hinv_v = v + (1 - damp) * hinv_v - hv / scale
            if verbose:
                tbar.update(1)
    else:
        if hinv is None:
            hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
        scale = 1.0
        hinv_v = np.matmul(hinv, v)

    return hinv_v / scale


clf = NeuralNetwork(input_size=X_train.shape[-1])
clf.fit(X_train, y_train)

y_pred_test = clf.predict_proba(X_test)
y_pred_train = clf.predict_proba(X_train)

spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
print("Initial statistical parity: ", spd_0)

tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
print("Initial TPR parity: ", tpr_parity_0)

predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
print("Initial predictive parity: ", predictive_parity_0)

loss_0 = logistic_loss(y_test, y_pred_test)
print("Initial loss: ", loss_0)

accuracy_0 = computeAccuracy(y_test, y_pred_test)
print("Initial accuracy: ", accuracy_0)

hessian_all_points = []
print('Computing Hessian')
tbar = tqdm.tqdm(total=len(X_train))
total_time = 0
for i in range(len(X_train)):
    t0 = time.time()
    hessian_all_points.append(hessian_one_point(clf, X_train[i], y_train[i]) / len(X_train))
    total_time += time.time() - t0
    tbar.update(1)

hessian_all_points = np.array(hessian_all_points)

del_L_del_theta = []
for i in range(int(len(X_train))):
    gradient = convert_grad_to_ndarray(del_L_del_theta_i(clf, X_train[i], int(y_train[i])))
    while np.sum(np.isnan(gradient)) > 0:
        gradient = convert_grad_to_ndarray(del_L_del_theta_i(clf, X_train[i], int(y_train[i])))
    del_L_del_theta.append(gradient)

metric = 0
if metric == 0:
    v1 = del_spd_del_theta(clf, X_test_orig, X_test, dataset)
elif metric == 1:
    v1 = del_tpr_parity_del_theta(clf, X_test_orig, X_test, y_test, dataset)
elif metric == 2:
    v1 = del_predictive_parity_del_theta(clf, X_test_orig, X_test, y_test, dataset)

hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
hinv_v = s_test(clf, X_train, y_train, v1, hinv=hinv, verbose=False)


def first_order_influence(del_L_del_theta, hinv_v, n):
    infs = []
    for i in range(n):
        inf = -np.dot(del_L_del_theta[i].transpose(), hinv_v)
        inf *= -1 / n
        infs.append(inf)
    return infs


def second_order_influence(model, X_train, y_train, U, del_L_del_theta, r=-1, verbose=False):
    u = len(U)
    s = len(X_train)
    p = u / s
    c1 = (1 - 2 * p) / (s * (1 - p) ** 2)
    c2 = 1 / ((s * (1 - p)) ** 2)
    num_params = len(del_L_del_theta[0])
    del_L_del_theta_sum = np.sum([del_L_del_theta[i] for i in U], axis=0)
    hinv_del_L_del_theta = s_test(model, X_train, y_train, del_L_del_theta_sum, hinv=hinv)
    hessian_U_hinv_del_L_del_theta = np.zeros((num_params,))
    for i in range(u):
        idx = U[i]
        x, y = torch.FloatTensor(X_train[idx]), torch.FloatTensor([y_train[idx]])
        loss = loss_func(model, x, y)
        params = [p for p in model.parameters() if p.requires_grad]
        hessian_U_hinv_del_L_del_theta += convert_grad_to_ndarray(
            hvp(loss, params, torch.FloatTensor(hinv_del_L_del_theta)))

    term1 = c1 * hinv_del_L_del_theta
    term2 = c2 * s_test(model, X_train, y_train, hessian_U_hinv_del_L_del_theta, hinv=hinv)
    sum_term = term1 + term2
    return sum_term


def first_order_group_influence(U, del_L_del_theta):
    infs = []
    u = len(U)
    n = len(X_train)
    for i in range(u):
        inf = -np.dot(del_L_del_theta[U[i]].transpose(), hinv)
        inf *= -1 / n
        infs.append(inf)
    return np.sum(infs, axis=0)


def second_order_group_influence(U, del_L_del_theta):
    u = len(U)
    s = len(X_train)
    p = u / s
    c1 = (1 - 2 * p) / (s * (1 - p) ** 2)
    c2 = 1 / ((s * (1 - p)) ** 2)
    del_L_del_theta_sum = np.sum([del_L_del_theta[i] for i in U], axis=0)
    hinv_del_L_del_theta = np.matmul(hinv, del_L_del_theta_sum)
    hessian_U_hinv_del_L_del_theta = np.sum(np.matmul(hessian_all_points[U, :], hinv_del_L_del_theta), axis=0)
    term1 = c1 * hinv_del_L_del_theta
    term2 = c2 * np.matmul(hinv, hessian_U_hinv_del_L_del_theta)
    sum_term = (term1 + term2 * len(X_train))
    return sum_term


def get_subset(explanation):
    subset = X_train_orig.copy()
    for predicate in explanation:
        attr = predicate.split("=")[0].strip(' ')
        val = int(predicate.split("=")[1].strip(' '))
        subset = subset[subset[attr] == val]
    return subset.index


infs_1 = first_order_influence(del_L_del_theta, hinv_v, len(X_train))

alpha_f_lower = (-0.01) * (spd_0)
alpha_f_upper = -spd_0
del_f_threshold = (0.1) * spd_0
support = 0.05  # Do not consider extremely small patterns
support_small = 0.3  # For small patterns, 2nd-order estimation is quite accurate
del_f_threshold_small = (-0.1) * (spd_0)
print("alpha_f_lower:", alpha_f_lower)
print("alpha_f_upper:", alpha_f_upper)
print("del_f_threshold:", del_f_threshold)
print("support_small:", support_small)
print("del_f_threshold_small:", del_f_threshold_small)

clf = NeuralNetwork(input_size=X_train.shape[-1])
clf.fit(X_train, y_train)

attributes = []
attributeValues = []
first_order_influences = []
second_order_influences = []
fractionRows = []

v1_orig = v1
for col in X_train_orig.columns:
    vals = X_train_orig[col].unique()
    for val in vals:
        idx = X_train_orig[X_train_orig[col] == val].index
        if len(idx) / len(X_train) > support:
            X = np.delete(X_train, idx, 0)
            y = y_train.drop(index=idx, inplace=False)
            if len(y.unique()) > 1:
                idx = X_train_orig[X_train_orig[col] == val].index

                # First-order subset influence
                del_f_1 = 0
                params_f_2 = second_order_group_influence(idx, del_L_del_theta)
                del_f_2 = np.dot(v1.transpose(), params_f_2)

                attributes.append(col)
                attributeValues.append(val)
                first_order_influences.append(del_f_1)
                second_order_influences.append(del_f_2)
                fractionRows.append(len(idx) / len(X_train) * 100)

expl = [attributes, attributeValues, first_order_influences, second_order_influences, fractionRows]
expl = np.array(expl).T.tolist()

explanations = pd.DataFrame(expl, columns=["attributes", "attributeValues", "first_order_influences",
                                           "second_order_influences", "fractionRows"])
explanations['second_order_influences'] = explanations['second_order_influences'].astype(float)
explanations['first_order_influences'] = explanations['first_order_influences'].astype(float)
explanations['fractionRows'] = explanations['fractionRows'].astype(float)

# candidates = explanations[(explanations["second_order_influences"] > alpha_f_lower)
#                            & (explanations["second_order_influences"] < alpha_f_upper)]
candidates = copy.deepcopy(explanations)
candidates.loc[:, 'score'] = candidates.loc[:, 'second_order_influences'] * 100 / candidates.loc[:, 'fractionRows']

candidates_all = []
t_total = []
t_1 = []
t_2 = []
total_rows = len(X_train_orig)

# Generating 1-candidates
t0 = time.time()
candidates_1 = []
for i in range(len(candidates)):
    candidate = []
    candidate_i = candidates.iloc[i]
    #     if ((candidate_i["second_order_influences"] > del_f_threshold) &
    #         (candidate_i["fractionRows"] > support)):
    if ((candidate_i["fractionRows"] >= support_small) or
            ((candidate_i["fractionRows"] >= support) & (candidate_i["second_order_influences"] > del_f_threshold))
    ):
        attr_i = candidate_i["attributes"]
        val_i = int(candidate_i["attributeValues"])
        idx = X_train_orig[X_train_orig[attr_i] == val_i].index
        predicates = [attr_i + '=' + str(val_i)]
        candidate = [predicates, candidate_i["fractionRows"],
                     candidate_i["score"], candidate_i["second_order_influences"], idx]
        candidates_1.append(candidate)

print("Generated: ", len(candidates_1), " 1-candidates")
candidates_1.sort()
# display(candidates_1)

for i in range(len(candidates_1)):
    if float(candidates_1[i][2]) >= support:  # if score > top-k, keep in candidates, not otherwise
        candidates_all.insert(len(candidates_all), candidates_1[i])

# Generating 2-candidates
candidates_2 = []
for i in range(len(candidates_1)):
    predicate_i = candidates_1[i][0][0]
    attr_i = predicate_i.split("=")[0]
    val_i = int(float(predicate_i.split("=")[1]))
    sup_i = candidates_1[i][1]
    idx_i = candidates_1[i][-1]
    for j in range(i):
        predicate_j = candidates_1[j][0][0]
        attr_j = predicate_j.split("=")[0]
        val_j = int(float(predicate_j.split("=")[1]))
        sup_j = candidates_1[j][1]
        idx_j = candidates_1[j][-1]
        if attr_i != attr_j:
            idx = idx_i.intersection(idx_j)
            #             print(idx_i, idx_j, idx)
            fractionRows = len(idx) / total_rows * 100
            isCompact = True
            if fractionRows == min(sup_i, sup_j):  # pattern is not compact if intersection equals one of its parents
                isCompact = False
            if fractionRows / 100 >= support:
                params_f_2 = second_order_group_influence(idx, del_L_del_theta)
                del_f_2 = np.dot(v1.transpose(), params_f_2)
                score = del_f_2 * 100 / fractionRows
                if ((fractionRows / 100 >= support_small) or
                        ((score > candidates_1[i][2]) & (score > candidates_1[j][2]))):
                    predicates = [attr_i + '=' + str(val_i), attr_j + '=' + str(val_j)]
                    candidate = [sorted(predicates, key=itemgetter(0)), len(idx) * 100 / total_rows,
                                 score, del_f_2, idx]
                    candidates_2.append(candidate)
                    #                         print(candidate)
                    if isCompact:
                        candidates_all.append(candidate)
print("Generated: ", len(candidates_2), " 2-candidates")

# Recursively generating the rest
candidates_L_1 = copy.deepcopy(candidates_2)
set_L_1 = set()
iteration = 2
while (len(candidates_L_1) > 0) & (iteration < 4):
    print("Generated: ", iteration)
    candidates_L = []
    for i in range(len(candidates_L_1)):
        candidate_i = set(candidates_L_1[i][0])
        sup_i = candidates_L_1[i][1]
        idx_i = candidates_L_1[i][-1]
        for j in range(i):
            candidate_j = set(candidates_L_1[j][0])
            sup_j = candidates_L_1[j][1]
            idx_j = candidates_L_1[j][-1]
            merged_candidate = sorted(candidate_i.union(candidate_j), key=itemgetter(0))
            if json.dumps(merged_candidate) in set_L_1:
                continue
            if len(merged_candidate) == iteration + 1:
                intersect_candidates = candidate_i.intersection(candidate_j)
                setminus_i = list(candidate_i - intersect_candidates)[0].split("=")
                setminus_j = list(candidate_j - intersect_candidates)[0].split("=")
                attr_i = setminus_i[0]
                val_i = int(setminus_i[1])
                attr_j = setminus_j[0]
                val_j = int(setminus_j[1])
                if attr_i != attr_j:
                    # merge to get L list
                    idx = idx_i.intersection(idx_j)
                    fractionRows = len(idx) / len(X_train) * 100
                    isCompact = True
                    if fractionRows == min(sup_i,
                                           sup_j):  # pattern is not compact if intersection equals one of its parents
                        isCompact = False
                    if fractionRows / 100 >= support:
                        X = np.delete(X_train, idx, 0)
                        y = y_train.drop(index=idx, inplace=False)

                        params_f_2 = second_order_group_influence(idx, del_L_del_theta)
                        del_f_2 = np.dot(v1.transpose(), params_f_2)

                        score = del_f_2 * 100 / fractionRows
                        if (((score > candidates_L_1[i][2]) & (score > candidates_L_1[j][2])) or
                                (fractionRows >= support_small)):
                            candidate = [merged_candidate, fractionRows,
                                         del_f_2 * len(X_train) / len(idx), del_f_2, idx]
                            candidates_L.append(candidate)
                            set_L_1.add(json.dumps(merged_candidate))
                            if isCompact:
                                candidates_all.insert(len(candidates_all), candidate)
                    t_total.append(time.time() - t0)
    set_L_1 = set()
    print("Generated:", len(candidates_L), " ", str(iteration + 1), "-candidates")
    candidates_L_1 = copy.deepcopy(candidates_L)
    candidates_L_1.sort()
    iteration += 1
print(f'total time cost: {time.time() - t0}')

candidates_support_3_compact = copy.deepcopy(candidates_all)
print(len(candidates_support_3_compact))
candidates_df_3_compact = pd.DataFrame(candidates_support_3_compact,
                                       columns=["predicates", "support", "score", "2nd-inf", 'idx'])
candidates_df_3_compact = candidates_df_3_compact.sort_values(by=['score'], ascending=False)
print(len(candidates_df_3_compact))


class Topk:
    """
        top explanations: explanation -> (minhash, set_index, score)
    """

    def __init__(self, method='containment', threshold=0.75, k=5):
        self.method = method
        if method == 'lshensemble':
            raise NotImplementedError
        elif method == 'lsh':
            raise NotImplementedError

        self.top_explanations = dict()
        self.k = k
        self.threshold = threshold
        self.min_score = -100
        self.min_score_explanation = None
        self.containment_hist = []

    def _update_min(self, new_explanation, new_score):
        if len(self.top_explanations) > 0:
            for explanation, t in self.top_explanations.items():
                if t[2] < new_score:
                    new_score = t[2]
                    new_explanation = explanation
        self.min_score = new_score
        self.min_score_explanation = new_explanation

    def _containment(self, x, q):
        c = len(x & q) / len(q)
        self.containment_hist.append(c)
        return c

    def update(self, explanation, score):
        if (len(self.top_explanations) < self.k) or (score > self.min_score):
            s = get_subset(explanation)
            explanation = json.dumps(explanation)

            if self.method == 'lshensemble':
                raise NotImplementedError
            elif self.method == 'lsh':
                raise NotImplementedError
            elif self.method == 'containment':
                q_result = set()
                for k, v in self.top_explanations.items():
                    if self._containment(v[0], s) > self.threshold:
                        q_result.add(k)

            if len(q_result) == 0:
                if len(self.top_explanations) <= self.k - 1:
                    self._update_min(explanation, score)
                    self.top_explanations[explanation] = (s, score)
                    return 0
        return -1


containment_df = candidates_df_3_compact[candidates_df_3_compact['support'] < 10].sort_values(by=['score'],
                                                                                              ascending=False).copy()

topk = Topk(method='containment', threshold=0.2, k=5)
for row_idx in range(len(containment_df)):
    row = containment_df.iloc[row_idx]
    explanation, score = row[0], row[2]
    topk.update(explanation, score)
    if len(topk.top_explanations) == topk.k:
        break

explanations = list(topk.top_explanations.keys())
idxs = [v[1] for v in topk.top_explanations.values()]
supports = list()
scores = list()
gt_scores = list()
infs = list()
gts = list()
new_accs = list()
for e in explanations:
    idx = get_subset(json.loads(e))
    X = np.delete(X_train, idx, 0)
    y = y_train.drop(index=idx, inplace=False)
    clf.fit(np.array(X), np.array(y))
    y_pred = clf.predict_proba(np.array(X_test))
    new_acc = computeAccuracy(y_test, y_pred)
    inf_gt = computeFairness(y_pred, X_test_orig, y_test, 0, dataset) - spd_0

    condition = candidates_df_3_compact.predicates.apply(lambda x: x == json.loads(e))
    supports.append(float(candidates_df_3_compact[condition]['support']))
    scores.append(float(candidates_df_3_compact[condition]['score']))
    infs.append(float(candidates_df_3_compact[condition]['2nd-inf']))
    gts.append(inf_gt / (-spd_0))
    gt_scores.append(inf_gt * 100 / float(candidates_df_3_compact[condition]['support']))
    new_accs.append(new_acc)

expl = [explanations, supports, scores, gt_scores, infs, gts, new_accs]
expl = np.array(expl).T.tolist()

explanations = pd.DataFrame(expl, columns=["explanations", "support", "score", "gt-score", "2nd-inf(%)", "gt-inf(%)",
                                           "new-acc"])
explanations['score'] = explanations['score'].astype(float)
explanations['gt-score'] = explanations['gt-score'].astype(float)
explanations['support'] = explanations['support'].astype(float)
explanations['2nd-inf(%)'] = explanations['2nd-inf(%)'].astype(float) / (-spd_0)
explanations['gt-inf(%)'] = explanations['gt-inf(%)'].astype(float)
explanations['new-acc'] = explanations['new-acc'].astype(float)

pd.set_option('max_colwidth', 100)
explanations.sort_values(by=['score'], ascending=False)
