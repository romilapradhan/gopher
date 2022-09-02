import torch
from torch.autograd import grad
import math
import numpy as np
import json

with open('config.json', 'r') as f:
    txt = f.read()
    dtype = json.loads(txt)['dtype']
    f.close()
if dtype == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)

c = 1e-8  # small constant


def logistic_loss(y_true, y_pred):
    loss = 0
    for i in range(len(y_true)):
        if y_pred[i] != 0 and y_pred[i] != 1:
            loss += - y_true[i] * math.log(y_pred[i] + c) - (1 - y_true[i]) * math.log(1 - y_pred[i] + c)
    loss /= len(y_true)
    return loss


def binary_cross_entropy(y_pred, y_true, sample_weights=None):
    if sample_weights is None:
        loss = -(torch.log(y_pred + c) * y_true + torch.log(1 - y_pred + c) * (1 - y_true))
    else:
        loss = (-(torch.log(y_pred + c) * y_true + torch.log(1 - y_pred + c) * (1 - y_true))) * sample_weights
    return loss.mean()


# logistic_loss_torch = torch.nn.functional.binary_cross_entropy
def logistic_loss_torch(model, x, y_true, sample_weights=None, reg=1):
    if isinstance(y_true, int) or isinstance(y_true, np.ndarray):
        y_pred = model(torch.Tensor(x))
        return binary_cross_entropy(y_pred, torch.Tensor([y_true]), sample_weights)
    else:
        y_pred = model(x)
        return reg * model.C * torch.norm(convert_grad_to_tensor(list(model.parameters())),
                                          p=2) ** 2 + binary_cross_entropy(y_pred, y_true, sample_weights)


def nn_loss_torch(model, x, y_true, sample_weights=None, reg=1):
    if isinstance(y_true, int) or isinstance(y_true, np.ndarray):
        y_pred = model(torch.Tensor(x))
        return binary_cross_entropy(y_pred, torch.Tensor([y_true]), sample_weights)
    else:
        y_pred = model(x)
        return reg * model.C * torch.norm(convert_grad_to_tensor(list(model.parameters())),
                                          p=2) ** 2 + binary_cross_entropy(y_pred, y_true, sample_weights)


def svm_loss_torch(model, x, y_true, sample_weights=None):
    if isinstance(y_true, int) or isinstance(y_true, float):
        y_true = torch.Tensor([y_true])
    else:
        y_true = torch.Tensor(y_true)
    y_true = torch.where(y_true == 1.0, 1, -1)
    if sample_weights is None:
        return model.C * torch.norm(convert_grad_to_tensor(list(model.lr.parameters())), p=2) ** 2 / 2 + torch.mean(
            model.smooth_hinge(1 - y_true * model.decision_function(x)))
    else:
        return model.C * torch.norm(convert_grad_to_tensor(list(model.lr.parameters())), p=2) ** 2 / 2 + torch.mean(
            model.smooth_hinge(1 - y_true * model.decision_function(x)) * sample_weights)


def convert_grad_to_ndarray(grad):
    grad_list = list(grad)
    grad_arr = None
    for i in range(len(grad_list)):
        next_params = grad_list[i]

        if isinstance(next_params, torch.Tensor):
            next_params = next_params.detach().squeeze().numpy()

        if len(next_params.shape) == 0:
            next_params = np.expand_dims(next_params, axis=0)

        if len(next_params.shape) > 1:
            next_params = convert_grad_to_ndarray(next_params)

        if grad_arr is None:
            grad_arr = next_params
        else:
            grad_arr = np.concatenate([grad_arr, next_params])

    return grad_arr


def convert_grad_to_tensor(grad):
    grad_list = list(grad)
    grad_arr = None
    for i in range(len(grad_list)):
        next_params = grad_list[i]

        if len(next_params.shape) == 0:
            next_params = next_params.unsqueeze(0)

        if len(next_params.shape) > 1:
            next_params = convert_grad_to_tensor(next_params)

        if grad_arr is None:
            grad_arr = next_params
        else:
            grad_arr = torch.cat([grad_arr, next_params])

    return grad_arr


def del_f_del_theta_i(model, x, retain_graph=False):
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(model(torch.Tensor(x)), w, retain_graph=retain_graph)


def get_subset(explanation, X_train_orig):
    subset = X_train_orig.copy()
    for predicate in explanation:
        attr = predicate.split("=")[0].strip(' ')
        val = int(predicate.split("=")[1].strip(' '))
        subset = subset[subset[attr] == val]
    return subset.index


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
                if t[1] < new_score:
                    new_score = t[1]
                    new_explanation = explanation
        self.min_score = new_score
        self.min_score_explanation = new_explanation

    def _containment(self, x, q):
        c = len(x & q) / len(q)
        self.containment_hist.append(c)
        return c

    def update(self, explanation, score, X_train_orig):
        if (len(self.top_explanations) < self.k) or (score > self.min_score):
            s = get_subset(explanation, X_train_orig)
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
