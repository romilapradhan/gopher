import torch
from torch.autograd import grad
import math
import numpy as np


c = 1e-8  # small constant
def logistic_loss(y_true, y_pred):
    loss = 0
    for i in range(len(y_true)):
        if (y_pred[i] != 0 and y_pred[i] != 1):
            loss += - y_true[i] * math.log(y_pred[i] + c) - (1 - y_true[i]) * math.log(1 - y_pred[i] + c)
    loss /= len(y_true)
    return loss

def binary_cross_entropy(y_pred, y_true):
    loss = -(torch.log(y_pred+c)*y_true + torch.log(1-y_pred+c)*(1-y_true))
    return loss.mean()

# logistic_loss_torch = torch.nn.functional.binary_cross_entropy
def logistic_loss_torch(model, x, y_true):
    y_pred = model(torch.FloatTensor(x))
    return binary_cross_entropy(y_pred, torch.FloatTensor([y_true]))

def svm_loss_torch(clf, x, y_true):
    w = convert_grad_to_tensor(list(clf.parameters()))
    return clf.C*torch.sum(w**2) + torch.mean(clf.smooth_hinge(1-torch.Tensor(y_true)*clf.decision_function(x)))

def convert_grad_to_ndarray(grad):
    grad_list = list(grad)
    grad_arr = None
    for i in range(len(grad_list)):
        next_params = grad_list[i]
        
        if isinstance(next_params, torch.Tensor):
            next_params = next_params.detach().squeeze().numpy()

        if len(next_params.shape)==0:
            next_params = np.expand_dims(next_params, axis=0)

        if len(next_params.shape)>1:
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

        if len(next_params.shape)==0:
            next_params = next_params.unsqueeze(0)

        if len(next_params.shape)>1:
            next_params = convert_grad_to_tensor(next_params)

        if grad_arr is None:
            grad_arr = next_params
        else:
            grad_arr = torch.cat([grad_arr, next_params])
            
    return grad_arr

def del_L_del_theta_i(model, x, y_true, retain_graph=False):
    loss = loss_func(model, x, y_true)
    w = [ p for p in model.parameters() if p.requires_grad ]
    return grad(loss, w, create_graph=True, retain_graph=retain_graph)

def del_f_del_theta_i(model, x, retain_graph=False):
    w = [ p for p in model.parameters() if p.requires_grad ]
    return grad(model(torch.FloatTensor(x)), w, retain_graph=retain_graph)