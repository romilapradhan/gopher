from utils import del_f_del_theta_i, convert_grad_to_ndarray
from torch.autograd import grad
import torch
import numpy as np
import json

with open('config.json', 'r') as f:
    txt = f.read()
    dtype = json.loads(txt)['dtype']
    f.close()
if dtype == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)

def computeAccuracy(y_true, y_pred):
    return np.sum((y_pred>0.5) == y_true)/len(y_pred)
    
def computeFairness(y_pred, X_test, y_test, metric, dataset): 
    fairnessMetric = 0
    if dataset == 'german':
        protected_idx = X_test[X_test['age']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['age']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'compas':
        protected_idx = X_test[X_test['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'adult':
        protected_idx = X_test[X_test['gender']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['gender']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'traffic':
        protected_idx = X_test[X_test['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'sqf':
        protected_idx = X_test[X_test['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'random':
        protected_idx = X_test[X_test['AA']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['AA']==1].index
        numPrivileged = len(privileged_idx)

    p_protected = 0
    for i in range(len(protected_idx)):
        p_protected += y_pred[protected_idx[i]]
    p_protected /= len(protected_idx)

    p_privileged = 0
    for i in range(len(privileged_idx)):
        p_privileged += y_pred[privileged_idx[i]]
    p_privileged /= len(privileged_idx)
    
    # statistical parity difference
    statistical_parity = p_protected - p_privileged
    
    # equality of opportunity, or 
    # true positive rate parity
    # P(Y=1 | Y=1, G=0)- P(Y=1 | Y=1, G=1)
    true_positive_protected = 0
    actual_positive_protected = 0
    for i in range(len(protected_idx)):
        if y_test[protected_idx[i]] == 1:
            actual_positive_protected += 1
            true_positive_protected += y_pred[protected_idx[i]]
    tpr_protected = true_positive_protected/actual_positive_protected

    true_positive_privileged = 0
    actual_positive_privileged = 0
    for i in range(len(privileged_idx)):
        if y_test[privileged_idx[i]] == 1:
            actual_positive_privileged += 1
#             if (y_pred[privileged_idx[i]][1] > y_pred[privileged_idx[i]][0]):
            true_positive_privileged += y_pred[privileged_idx[i]]
    tpr_privileged = true_positive_privileged/actual_positive_privileged

    tpr_parity = tpr_protected - tpr_privileged
    
    # equalized odds or TPR parity + FPR parity
    # false positive rate parity
    
    # predictive parity
    p_o1_y1_s1 = 0
    p_o1_s1 = 0
    for i in range(len(protected_idx)):
#         if (y_pred[protected_idx[i]][1] > y_pred[protected_idx[i]][0]):
        p_o1_s1 += y_pred[protected_idx[i]]
        if y_test[protected_idx[i]] == 1:
            p_o1_y1_s1 += y_pred[protected_idx[i]]
    ppv_protected = p_o1_y1_s1/p_o1_s1
    
    p_o1_y1_s0 = 0
    p_o1_s0 = 0
    for i in range(len(privileged_idx)):
#         if (y_pred[privileged_idx[i]][1] > y_pred[privileged_idx[i]][0]):
        p_o1_s0 += y_pred[privileged_idx[i]]
        if y_test[privileged_idx[i]] == 1:
            p_o1_y1_s0 += y_pred[privileged_idx[i]]
    ppv_privileged = p_o1_y1_s0/p_o1_s0
    
    predictive_parity = ppv_protected - ppv_privileged
    
    if metric == 0:
        fairnessMetric = statistical_parity
    elif metric == 1:
        fairnessMetric = tpr_parity
    elif metric == 2:
        fairnessMetric = predictive_parity
        
    return fairnessMetric


def del_spd_del_theta(model, X_test_orig, X_test, dataset):
    num_params = len(convert_grad_to_ndarray(list(model.parameters())))
    del_f_protected = np.zeros((num_params,))
    del_f_privileged = np.zeros((num_params,))
    if dataset == 'german':
        numPrivileged = X_test_orig['age'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'compas':
        numPrivileged = X_test_orig['race'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'adult':
        numPrivileged = X_test_orig['gender'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'traffic':
        numPrivileged = X_test_orig['race'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'sqf':
        numPrivileged = X_test_orig['race'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'random':
        numPrivileged = X_test_orig['AA'].sum()
        numProtected = len(X_test_orig) - numPrivileged

    for i in range(len(X_test)):
        del_f_i = del_f_del_theta_i(model, X_test[i])
        del_f_i_arr = convert_grad_to_ndarray(del_f_i)
        if dataset == 'german':
            if X_test_orig.iloc[i]['age'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['age'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'compas':
            if X_test_orig.iloc[i]['race'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['race'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'adult':
            if X_test_orig.iloc[i]['gender'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['gender'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'traffic':
            if X_test_orig.iloc[i]['race'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['race'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'sqf':
            if X_test_orig.iloc[i]['race'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['race'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'random':
            if X_test_orig.iloc[i]['AA'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['AA'] == 0:
                del_f_protected += del_f_i_arr

    del_f_privileged /= numPrivileged
    del_f_protected /= numProtected
    v = del_f_protected - del_f_privileged
    return v


def del_tpr_parity_del_theta(model, X_test_orig, X_test, y_test, dataset):
    num_params = len(convert_grad_to_ndarray(list(model.parameters())))
    del_f_protected = np.zeros((num_params,))
    del_f_privileged = np.zeros((num_params,))
    
    if dataset == 'german':
        protected_idx = X_test_orig[X_test_orig['age']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['age']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'compas':
        protected_idx = X_test_orig[X_test_orig['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'adult':
        protected_idx = X_test_orig[X_test_orig['gender']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['gender']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'traffic':
        protected_idx = X_test_orig[X_test_orig['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'sqf':
        protected_idx = X_test_orig[X_test_orig['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'random':
        protected_idx = X_test_orig[X_test_orig['AA']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['AA']==1].index
        numPrivileged = len(privileged_idx)

    actual_positive_privileged = 0
    for i in range(len(privileged_idx)):
        if y_test[privileged_idx[i]] == 1:
            actual_positive_privileged += 1
            del_f_i = del_f_del_theta_i(model, X_test[privileged_idx[i]])
            del_f_i_arr = convert_grad_to_ndarray(del_f_i)
            del_f_privileged = np.add(del_f_privileged, del_f_i_arr)
    del_f_privileged /= actual_positive_privileged
    
    actual_positive_protected = 0
    for i in range(len(protected_idx)):
        if y_test[protected_idx[i]] == 1:
            actual_positive_protected += 1
            del_f_i = del_f_del_theta_i(model, X_test[protected_idx[i]])
            del_f_i_arr = convert_grad_to_ndarray(del_f_i)
            del_f_protected = np.add(del_f_protected, del_f_i_arr)
    del_f_protected /= actual_positive_protected

    v = del_f_protected - del_f_privileged
    return v


def del_predictive_parity_del_theta(model, X_test_orig, X_test, y_test, dataset):
    y_pred = model.predict_proba(X_test)
    num_params = len(convert_grad_to_ndarray(list(model.parameters())))
    del_f_protected = np.zeros((num_params, 1))
    del_f_privileged = np.zeros((num_params, 1))
    
    if dataset == 'german':
        protected_idx = X_test_orig[X_test_orig['age']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['age']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'compas':
        protected_idx = X_test_orig[X_test_orig['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'adult':
        protected_idx = X_test_orig[X_test_orig['gender']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['gender']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'traffic':
        protected_idx = X_test_orig[X_test_orig['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'sqf':
        protected_idx = X_test_orig[X_test_orig['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'random':
        protected_idx = X_test_orig[X_test_orig['AA']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test_orig[X_test_orig['AA']==1].index
        numPrivileged = len(privileged_idx)

    u_dash_protected = np.zeros((num_params,))
    v_protected = 0
    v_dash_protected = np.zeros((num_params,))
    u_protected = 0
    for i in range(len(protected_idx)):
        del_f_i = del_f_del_theta_i(model, X_test[protected_idx[i]])
        del_f_i_arr = convert_grad_to_ndarray(del_f_i)
        v_protected += y_pred[protected_idx[i]]
        v_dash_protected = np.add(v_dash_protected, del_f_i_arr)
        if y_test[protected_idx[i]] == 1:
            u_dash_protected = np.add(u_dash_protected, del_f_i_arr)
            u_protected += y_pred[protected_idx[i]]
    del_f_protected = (u_dash_protected * v_protected - u_protected * v_dash_protected)/(v_protected * v_protected)
    
    u_dash_privileged = np.zeros((num_params,))
    v_privileged = 0
    v_dash_privileged = np.zeros((num_params,))
    u_privileged = 0
    for i in range(len(privileged_idx)):
        del_f_i = del_f_del_theta_i(model, X_test[privileged_idx[i]])
        del_f_i_arr = convert_grad_to_ndarray(del_f_i)
        v_privileged += y_pred[privileged_idx[i]]
        v_dash_privileged = np.add(v_dash_privileged, del_f_i_arr)
        if y_test[privileged_idx[i]] == 1:
            u_dash_privileged = np.add(u_dash_privileged, del_f_i_arr)
            u_privileged += y_pred[privileged_idx[i]]
    del_f_privileged = (u_dash_privileged * v_privileged - u_privileged * v_dash_privileged)/(v_privileged * v_privileged)

    v = np.subtract(del_f_protected, del_f_privileged)
    return v