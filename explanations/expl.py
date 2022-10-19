import copy
from operator import itemgetter

import pandas as pd
from scipy.optimize import Bounds, minimize

from influence import *
from utils import *
from metrics import *


def explanation_candidate_generation(X_train_orig, X_train, y_train, dataset, del_L_del_theta,
                                     hessian_all_points, hinv, del_F_del_theta,
                                     del_f_threshold=0.01, support=0.05, support_small=0.3):
    """
    X_train_orig: the dataframe corresponding to X_train. X_train refers to the standardized training
     data in the format of ndarray.

    del_f_threshold: influence lower bound for the first-level patterns. The first-level patterns with influence
     lower than this value would be filtered out

    support: support lower bound for patterns. Patterns that are too small are not interesting as they are
     not representative. Thus all output pattern candidates should have support no lower than this value.

    support_small: support upper bound for patterns. Patterns that are too large are not interesting
     especially for removal. Thus all final pattern should have support no larger than this value.
    """
    attributes = []
    attributeValues = []
    second_order_influences = []
    fractionRows = []

    for col in X_train_orig.columns:
        if dataset == 'german':
            if "purpose" in col or "housing" in col:  # dummy variables purpose=0 doesn't make sense
                vals = [1]
            else:
                vals = X_train_orig[col].unique()
        elif dataset == 'adult':
            vals = X_train_orig[col].unique()
        elif dataset == 'compas':
            vals = X_train_orig[col].unique()
        elif dataset == 'sqf':
            vals = X_train_orig[col].unique()
        elif dataset == 'random':
            vals = X_train_orig[col].unique()
        else:
            raise NotImplementedError
        for val in vals:
            idx = X_train_orig[X_train_orig[col] == val].index
            if len(idx) / len(X_train) > support:
                y = y_train.drop(index=idx, inplace=False)
                if len(y.unique()) > 1:
                    idx = X_train_orig[X_train_orig[col] == val].index

                    # Second-order subset influence
                    params_f_2 = second_order_group_influence(idx, del_L_del_theta, hessian_all_points, hinv)
                    del_f_2 = np.dot(del_F_del_theta.transpose(), params_f_2)

                    attributes.append(col)
                    attributeValues.append(val)
                    second_order_influences.append(del_f_2)
                    fractionRows.append(len(idx) / len(X_train) * 100)

    expl = [attributes, attributeValues, second_order_influences, fractionRows]
    expl = np.array(expl).T.tolist()

    explanations = pd.DataFrame(expl,
                                columns=["attributes", "attributeValues", "second_order_influences", "fractionRows"])
    explanations['second_order_influences'] = explanations['second_order_influences'].astype(float)
    explanations['fractionRows'] = explanations['fractionRows'].astype(float)

    candidates = copy.deepcopy(explanations)
    candidates.loc[:, 'score'] = candidates.loc[:, 'second_order_influences'] * 100 / candidates.loc[:, 'fractionRows']

    candidates_all = []
    total_rows = len(X_train_orig)

    # Lattice-based search
    # Generating 1-candidates
    candidates_1 = []
    for i in range(len(candidates)):
        candidate_i = candidates.iloc[i]
        if ((candidate_i["fractionRows"] >= support_small) or ((candidate_i["fractionRows"] >= support) & (
                candidate_i["second_order_influences"] > del_f_threshold))):
            attr_i = candidate_i["attributes"]
            val_i = int(float(candidate_i["attributeValues"]))
            idx = X_train_orig[X_train_orig[attr_i] == val_i].index
            predicates = [attr_i + '=' + str(val_i)]
            candidate = [predicates, candidate_i["fractionRows"],
                         candidate_i["score"], candidate_i["second_order_influences"], idx]
            candidates_1.append(candidate)

    print("Generated: ", len(candidates_1), " 1-candidates")
    candidates_1.sort()

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
                fractionRows = len(idx) / total_rows * 100
                isCompact = True
                # pattern is not compact if intersection equals one of its parents
                if fractionRows == min(sup_i, sup_j):
                    isCompact = False
                if fractionRows / 100 >= support:
                    params_f_2 = second_order_group_influence(idx, del_L_del_theta, hessian_all_points, hinv)
                    del_f_2 = np.dot(del_F_del_theta.transpose(), params_f_2)
                    score = del_f_2 * 100 / fractionRows
                    if ((fractionRows / 100 >= support_small) or
                            ((score > candidates_1[i][2]) & (score > candidates_1[j][2]))):
                        predicates = [attr_i + '=' + str(val_i), attr_j + '=' + str(val_j)]
                        candidate = [sorted(predicates, key=itemgetter(0)), len(idx) * 100 / total_rows,
                                     score, del_f_2, idx]
                        candidates_2.append(candidate)
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
                    attr_j = setminus_j[0]
                    if attr_i != attr_j:
                        # merge to get L list
                        idx = idx_i.intersection(idx_j)
                        fractionRows = len(idx) / len(X_train) * 100
                        isCompact = True
                        # pattern is not compact if intersection equals one of its parents
                        if fractionRows == min(sup_i, sup_j):
                            isCompact = False
                        if fractionRows / 100 >= support:
                            params_f_2 = second_order_group_influence(idx, del_L_del_theta, hessian_all_points, hinv)
                            del_f_2 = np.dot(del_F_del_theta.transpose(), params_f_2)

                            score = del_f_2 * 100 / fractionRows
                            if (((score > candidates_L_1[i][2]) & (score > candidates_L_1[j][2])) or
                                    (fractionRows >= support_small)):
                                candidate = [merged_candidate, fractionRows,
                                             del_f_2 * len(X_train) / len(idx), del_f_2, idx]
                                candidates_L.append(candidate)
                                set_L_1.add(json.dumps(merged_candidate))
                                if isCompact:
                                    candidates_all.insert(len(candidates_all), candidate)
        set_L_1 = set()
        print("Generated:", len(candidates_L), " ", str(iteration + 1), "-candidates")
        candidates_L_1 = copy.deepcopy(candidates_L)
        candidates_L_1.sort()
        iteration += 1

    candidates_support_3_compact = copy.deepcopy(candidates_all)
    candidates_df_3_compact = pd.DataFrame(candidates_support_3_compact,
                                           columns=["predicates", "support", "score", "2nd-inf", 'idx'])
    candidates_df_3_compact = candidates_df_3_compact.sort_values(by=['score'], ascending=False)

    # thresholding
    containment_df = candidates_df_3_compact[candidates_df_3_compact['support'] < support_small * 100].sort_values(
        by=['score'], ascending=False).copy()
    print('# candidates left: ', len(containment_df))
    return containment_df


def get_top_k_expl(expl_df, X_train_orig, containment_threshold=0.2, k=5):
    """
    X_train_orig: dataframe, serve as an attr-index mapping
    """
    topk = Topk(method='containment', threshold=containment_threshold, k=k)
    for row_idx in range(len(expl_df)):
        row = expl_df.iloc[row_idx]
        explanation, score = row[0], row[2]
        topk.update(explanation, score, X_train_orig)
        if len(topk.top_explanations) == topk.k:
            break
    return topk


def get_update_expl(model, step_size, pattern_idx, X_train_orig, X_train, y_train, del_F_del_theta, loss_func):
    """
    X_train_orig: the dataframe corresponding to X_train. X_train refers to the standardized training
     data in the format of ndarray.

    del_F_del_theta: first-order derivative of the fairness metric w.r.t. model parameters. Can be calculated by
     get_del_F_del_theta in influence.py

    loss_func: loss function implemented using torch and support taking derivatives.
    """
    S = torch.Tensor(X_train[pattern_idx])
    S.requires_grad = True
    delta = torch.zeros(1, X_train.shape[-1])
    delta.requires_grad = True
    S_new = S + delta

    part_1 = torch.FloatTensor(del_F_del_theta).repeat(len(S_new), 1).reshape(len(S_new), 1, -1)
    part_2 = []
    for i in range(len(S_new)):
        inner_lst = []
        del_L_del_theta_i_t = convert_grad_to_tensor(del_L_del_theta_i(model, S_new[i], y_train[i],
                                                                       loss_func, retain_graph=True))
        for j in range(len(del_L_del_theta_i_t)):
            inner_grad = convert_grad_to_ndarray(grad(del_L_del_theta_i_t[j], delta, retain_graph=True))
            inner_lst.append(inner_grad)
        part_2.append(np.array(inner_lst))

    part_2 = np.array(part_2)
    part_2 = torch.FloatTensor(part_2)
    part_2 = part_2.mean(dim=0).unsqueeze(0).repeat(len(S_new), 1, 1)

    final = torch.bmm(part_1, part_2).reshape((len(S_new), -1))
    delta = delta - step_size * final  # single-step gradient descent
    S_new = S_new + delta

    X_train_new = X_train.copy()
    X_train_new[pattern_idx] = X_train_new[pattern_idx] + delta.detach().numpy()

    mins = []
    maxs = []
    numCols = len(X_train[0])
    for i in range(numCols):
        mins.insert(i, min(X_train[:, i]))
        maxs.insert(i, max(X_train[:, i]))

    bounds = Bounds(mins, maxs)
    tbar = tqdm.tqdm(total=len(S_new))
    for i in X_train_orig[pattern_idx].index:
        X_train_pert_pt = X_train_new[i]
        f = lambda x: np.linalg.norm(x - X_train_pert_pt)
        x0 = np.random.rand(numCols)
        res = minimize(f, x0, method='trust-constr', options={'verbose': 0}, bounds=bounds)
        X_train_new[i] = res.x
        tbar.update(1)

    return X_train_new
