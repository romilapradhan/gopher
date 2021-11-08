from sklearn.linear_model import SGDClassifier
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


class LogisticRegression(nn.Module):
    def __init__(self, input_size, learning_rate=0.05, c=0.03, epoch_num=100):
        super(LogisticRegression, self).__init__()
        self.sklearn_lr = SGDClassifier(loss='log', warm_start=True, max_iter=epoch_num,
                                        average=True, shuffle=False, learning_rate='constant',
                                        eta0=learning_rate, alpha=c, verbose=0)
        self.lr = torch.nn.Linear(input_size, 1, bias=True)
        self.sm = torch.nn.Sigmoid()
        self.C = c
        self.epoch_num = epoch_num
        self.criterion = logistic_loss_torch
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x.squeeze()

    def fit(self, x, y, verbose=False, use_sklearn=False):
        if use_sklearn:
            self.sklearn_lr.fit(x, y)
            self.C = self.sklearn_lr.C
            self.lr.weight.data = torch.Tensor(self.sklearn_lr.coef_)
            self.lr.bias.data = torch.Tensor(self.sklearn_lr.intercept_)

        else:
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            self.train()
            for _ in range(self.epoch_num):
                loss = self.criterion(self, x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()

    def load_weights_from_another_model(self, orig_model):
        self.C = orig_model.C
        self.lr.weight.data = orig_model.lr.weight.data.clone()
        self.lr.bias.data = orig_model.lr.bias.data.clone()

    def partial_fit(self, x, y, learning_rate=0.05):
        params = {'learning_rate': 'constant', 'eta0': learning_rate}
        self.sklearn_lr.set_params(**params)
        self.sklearn_lr.partial_fit(x, y, classes=y.unique())
        self.lr.weight.data = torch.Tensor(self.sklearn_lr.coef_)


class SVM(nn.Module):
    def __init__(self, input_size, learning_rate=0.05, c=0.1, epoch_num=100, kernel='linear'):
        super(SVM, self).__init__()
        self.sklearn_svc = SGDClassifier(random_state=0, warm_start=True, max_iter=epoch_num,
                                         average=True, shuffle=False, learning_rate='constant',
                                         eta0=learning_rate, alpha=c, loss='hinge')
        self.lr = torch.nn.Linear(input_size, 1, bias=True)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.8)
        self.smooth_hinge = torch.nn.Softplus(beta=1)
        self.C = c
        self.epoch_num = epoch_num
        if kernel != 'linear':
            raise NotImplementedError

    def decision_function(self, x):
        if ~isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.lr(x)
        return x.squeeze()

    def forward(self, x):
        if ~isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.lr(x)
        x = 1 / (1 + torch.exp(-x))
        return x.squeeze()

    def fit(self, x, y, use_sklearn=False):
        if use_sklearn:
            self.sklearn_svc.fit(x, y)
            self.C = self.sklearn_svc.C
            self.lr.weight.data = torch.Tensor(self.sklearn_svc.coef_)
            self.lr.bias.data = torch.Tensor(self.sklearn_svc.intercept_)
        else:
            criterion = svm_loss_torch
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            for _ in range(self.epoch_num):
                loss = criterion(self, x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.Tensor(x)).detach().numpy()
    
    def partial_fit(self, x, y, learning_rate=0.05):
        params = {'learning_rate': 'constant', 'eta0': learning_rate}
        self.sklearn_svc.set_params(**params)
        self.sklearn_svc.partial_fit(x, y, classes=y.unique())
        self.lr.weight.data = torch.Tensor(self.sklearn_svc.coef_)
        self.lr.bias.data = torch.Tensor(self.sklearn_svc.intercept_)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, learning_rate=0.05, c=0.01, epoch_num=1000, batch_size=80):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 10)
        self.sm1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(10, 1)
        self.sm2 = torch.nn.Sigmoid()
        self.input_size = input_size
        self.C = c
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.criterion = binary_cross_entropy
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=c, momentum=0.9)
        # best result according to grid search
        self.sklearn_nn = MLPClassifier(random_state=0, alpha=c, learning_rate='adaptive', batch_size=batch_size,
                                        solver='adam', hidden_layer_sizes=(10,), activation='logistic')

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.sm1(x)
        x = self.fc2(x)
        x = self.sm2(x)
        return x.squeeze()

    def fit(self, x, y, use_sklearn=False):
        if use_sklearn:
            self.sklearn_nn.fit(x, y)
            self.fc1.weight.data = torch.Tensor(self.sklearn_nn.coefs_[0]).T
            self.fc1.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[0]).T
            self.fc2.weight.data = torch.Tensor(self.sklearn_nn.coefs_[1]).T
            self.fc2.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[1]).T
        else:
            num_batches = len(x)//self.batch_size+1 if len(x)%self.batch_size!=0 else len(x)//self.batch_size
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            for _ in range(self.epoch_num):
                for batch_id in range(num_batches):
                    if batch_id < num_batches-1:
                        x_ = x[batch_id*self.batch_size:(batch_id+1)*self.batch_size]
                        y_ = y[batch_id*self.batch_size:(batch_id+1)*self.batch_size]
                    else:
                        x_ = x[batch_id*self.batch_size:]
                        y_ = y[batch_id*self.batch_size:]
                    y_pred = self.forward(x_)
                    loss = self.criterion(y_pred, y_)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def predict_proba(self, x):
        self.eval()
        x = torch.Tensor(x).view(-1, self.input_size)
        return self.forward(x).detach().numpy()
    
    def partial_fit(self, x, y, learning_rate=0.05):
        params = {'learning_rate': 'constant', 'learning_rate_init': learning_rate}
        self.sklearn_nn.set_params(**params)
        self.sklearn_nn.partial_fit(x, y, classes=y.unique())
        self.fc1.weight.data = torch.Tensor(self.sklearn_nn.coefs_[0]).T
        self.fc1.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[0]).T
        self.fc2.weight.data = torch.Tensor(self.sklearn_nn.coefs_[1]).T
        self.fc2.bias.data = torch.Tensor(self.sklearn_nn.intercepts_[1]).T