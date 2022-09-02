import numpy as np
from utils import *
from metrics import *
import tqdm


def second_order_group_influence(U, del_L_del_theta, hessian_all_points, hinv):
    u = len(U)
    s = len(del_L_del_theta)
    p = u/s
    c1 = (1 - 2*p)/(s * (1-p)**2)
    c2 = 1/((s * (1-p))**2)
    del_L_del_theta_sum = np.sum(del_L_del_theta[U, :], axis=0)
    hinv_del_L_del_theta = np.matmul(hinv, del_L_del_theta_sum)
    hessian_U_hinv_del_L_del_theta = np.sum(np.matmul(hessian_all_points[U, :], hinv_del_L_del_theta), axis=0)
    term1 = c1 * hinv_del_L_del_theta
    term2 = c2 * np.matmul(hinv, hessian_U_hinv_del_L_del_theta)
    sum_term = (term1 + term2*len(del_L_del_theta))
    return sum_term


def get_del_F_del_theta(model, X_test_orig, X_test, y_test, dataset, metric):
    if metric == 0:
        v1 = del_spd_del_theta(model, X_test_orig, X_test, dataset)
    elif metric == 1:
        v1 = del_tpr_parity_del_theta(model, X_test_orig, X_test, y_test, dataset)
    elif metric == 2:
        v1 = del_predictive_parity_del_theta(model, X_test_orig, X_test, y_test, dataset)
    else:
        raise NotImplementedError
    return v1


def del_L_del_theta_i(model, x, y_true, loss_func, retain_graph=False):
    loss = loss_func(model, x, y_true)
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(loss, w, create_graph=True, retain_graph=retain_graph)


def get_del_L_del_theta(model, X_train, y_train, loss_func):
    del_L_del_theta = []
    for i in range(int(len(X_train))):
        gradient = convert_grad_to_ndarray(del_L_del_theta_i(model, X_train[i], int(y_train[i]), loss_func))
        while np.sum(np.isnan(gradient))>0:
            gradient = convert_grad_to_ndarray(del_L_del_theta_i(model, X_train[i], int(y_train[i]), loss_func))
        del_L_del_theta.append(gradient)
    del_L_del_theta = np.array(del_L_del_theta)
    return del_L_del_theta


def hvp(y, w, v):
    ''' Multiply the Hessians of y and w by v.'''
    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(convert_grad_to_tensor(first_grads), v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def hessian_one_point(model, x, y, loss_func):
    x, y = torch.FloatTensor(x), torch.FloatTensor([y])
    loss = loss_func(model, x, y)
    params = [ p for p in model.parameters() if p.requires_grad ]
    first_grads = convert_grad_to_tensor(grad(loss, params, retain_graph=True, create_graph=True))
    hv = np.zeros((len(first_grads), len(first_grads)))
    for i in range(len(first_grads)):
        hv[i, :] = convert_grad_to_ndarray(grad(first_grads[i], params, create_graph=True)).ravel()
    return hv


def get_hessian_all_points(model, X_train, y_train, loss_func):
    hessian_all_points = []
    tbar = tqdm.tqdm(total=len(X_train))
    for i in range(len(X_train)):
        hessian_all_points.append(hessian_one_point(model, X_train[i], y_train[i], loss_func)/len(X_train))
        tbar.update(1)
    hessian_all_points = np.array(hessian_all_points)
    return hessian_all_points


# Compute multiplication of inverse hessian matrix and vector v
def get_hinv_v(hessian_all_points, v, hinv=None, recursive=False):
    if recursive:
        raise NotImplementedError
    else:
        if hinv is None:
            hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
        hinv_v = np.matmul(hinv, v)

    return hinv_v, hinv