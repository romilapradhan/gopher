from sklearn.linear_model import LogisticRegression as sklr
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPClassifier
from utils import *
import sklearn
import numpy as np
import torch.nn as nn
import torch
import json
import time

with open('config.json', 'r') as f:
    txt = f.read()
    dtype = json.loads(txt)['dtype']
    f.close()
if dtype == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


def binary_cross_entropy(y_pred, y_true):
    loss = -(torch.log(y_pred + c) * y_true + torch.log(1 - y_pred + c) * (1 - y_true))
    return loss.mean()


class LogisticRegression(nn.Module):
    def __init__(self, input_size, learning_rate=0.05, c=0.03, epoch_num=100):
        super(LogisticRegression, self).__init__()
        # self.sklearn_lr = sklearn.linear_model.SGDClassifier(loss='log', warm_start=True, max_iter=epoch_num,
        # average=True, shuffle=False, learning_rate='constant', eta0=learning_rate, tol=0, alpha=c, n_jobs=1,
        # early_stopping=False, verbose=0)
        self.sklearn_lr = sklearn.linear_model.SGDClassifier(loss='log', warm_start=True, max_iter=epoch_num,
                                                             average=True, shuffle=False, learning_rate='constant',
                                                             eta0=learning_rate, alpha=c, verbose=0)
        #         self.sklearn_lr = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=100, solver='sag')
        self.lr = torch.nn.Linear(input_size, 1, bias=True)
        self.sm = torch.nn.Sigmoid()
        self.C = c
        self.epoch_num = epoch_num
        self.criterion = binary_cross_entropy
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=c)

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x.squeeze()

    def fit(self, x, y, verbose=False, use_sklearn=True):
        if use_sklearn:
            self.sklearn_lr.fit(x, y)
            #             classes = np.unique(y)
            #             for _ in range(epoch_num):
            #                 self.sklearn_lr.partial_fit(x ,y, classes)
            self.C = self.sklearn_lr.C
            self.lr.weight.data = torch.Tensor(self.sklearn_lr.coef_)
            self.lr.bias.data = torch.Tensor(self.sklearn_lr.intercept_)

        else:
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            self.train()
            for _ in range(self.epoch_num):
                y_pred = self.forward(x)
                loss = self.criterion(y_pred, y)
                #                 l2_reg = torch.norm(self.lr.weight, p=2)**2
                #                 loss += c * l2_reg
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        #                 scheduler.step()
        #                 print(self.criterion(y_pred, y))
        # if verbose and (epoch_num % 50):
        #     print(f'epoch:{epoch_num}, loss:{loss.item()}')

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()

    def load_weights_from_another_model(self, orig_model):
        self.C = orig_model.C
        self.lr.weight.data = orig_model.lr.weight.data.clone()
        self.lr.bias.data = orig_model.lr.bias.data.clone()

    def partial_fit(self, x, y, learning_rate=0.05):
        default_params = {'learning_rate': 'optimal', 'eta0': 0.0}
        params = {'learning_rate': 'constant', 'eta0': learning_rate}
        self.sklearn_lr.set_params(**params)
        self.sklearn_lr.partial_fit(x, y, classes=y.unique())
        self.C = self.sklearn_lr.C
        self.lr.weight.data = torch.Tensor(self.sklearn_lr.coef_)
        self.sklearn_lr.set_params(**default_params)


class SVM(nn.Module):
    def __init__(self, input_size, kernel='linear'):
        super(SVM, self).__init__()
        self.sklearn_svc = LinearSVC(random_state=0, loss='hinge')
        self.lr = torch.nn.Linear(input_size, 1, bias=True)
        self.initialize_weights(self.lr)
        self.smooth_hinge = torch.nn.Softplus(beta=0.001)
        #         self.smooth_hinge = torch.nn.ReLU()
        if kernel != 'linear':
            raise NotImplementedError

    def initialize_weights(self, m):
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)

    def decision_function(self, x):
        if ~isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.lr(x)
        return x.squeeze()

    def forward(self, x):
        if ~isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.lr(x)
        x = 1 - 1 / (1 + torch.exp(x))
        return x.squeeze()

    def fit(self, x, y, c=1.0, epoch_num=1000, verbose=False, use_sklearn=False):
        if use_sklearn:
            self.sklearn_svc.fit(x, y)
            self.C = self.sklearn_svc.C
            self.lr.weight.data = torch.Tensor(self.sklearn_svc.coef_)
            self.lr.bias.data = torch.Tensor(self.sklearn_svc.intercept_)
        else:
            criterion = svm_loss_torch
            self.C = c
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            self.train()
            for _ in range(epoch_num):
                loss = criterion(self, x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if verbose and (epoch_num % 50):
            print(f'epoch:{epoch_num}, loss:{loss.item()}')

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 20)
        self.sm1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(20, 1)
        self.sm2 = torch.nn.Sigmoid()
        self.input_size = input_size

        # best result according to grid search
        self.sklearn_nn = MLPClassifier(random_state=0, alpha=0.01, learning_rate='adaptive', batch_size=1024,
                                        solver='adam', hidden_layer_sizes=(20,), activation='logistic')

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.sm1(x)
        x = self.fc2(x)
        x = self.sm2(x)
        return x.squeeze()

    def fit(self, x, y):
        self.sklearn_nn.fit(x, y)
        self.fc1.weight.data = torch.Tensor(self.sklearn_nn.coefs_[0]).T
        self.fc1.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[0]).T
        self.fc2.weight.data = torch.Tensor(self.sklearn_nn.coefs_[1]).T
        self.fc2.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[1]).T

    def predict_proba(self, x):
        self.eval()
        x = torch.Tensor(x).view(-1, self.input_size)
        return self.forward(x).detach().numpy()
