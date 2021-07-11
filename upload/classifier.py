from sklearn.linear_model import LogisticRegression as sklr
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPClassifier
from utils import *
import numpy as np
import torch.nn as nn
import torch


def binary_cross_entropy(y_pred, y_true):
    loss = -(torch.log(y_pred+c)*y_true + torch.log(1-y_pred+c)*(1-y_true))
    return loss.mean()

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.sklearn_lr = sklr(random_state=0, max_iter=300)
        self.lr = torch.nn.Linear(input_size, 1, bias=True)
        self.initialize_weights(self.lr)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x.squeeze()
    
    def fit(self, x, y, c=0.2, epoch_num=1500, verbose=False, use_sklearn=False):
        if use_sklearn:
            self.sklearn_lr.fit(x, y)
            self.C = self.sklearn_lr.C
            self.lr.weight.data = torch.FloatTensor(self.sklearn_lr.coef_)
            self.lr.bias.data = torch.FloatTensor(self.sklearn_lr.intercept_)
        else:
            self.C = c
            criterion = logistic_loss_torch
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y)
            self.train()
            for _ in range(epoch_num):
                loss = criterion(self, x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if verbose and (epoch_num % 50):
            print(f'epoch:{epoch_num}, loss:{loss.item()}')
    
    def initialize_weights(self, m):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0)

    def predict_proba(self, x):
        self.eval()
        return self.forward(torch.FloatTensor(x)).detach().numpy()
    
    def load_weights_from_another_model(self, orig_model):
        self.C = orig_model.C
        self.lr.weight.data = orig_model.lr.weight.data.clone()
        self.lr.bias.data = orig_model.lr.bias.data.clone()


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
            x = torch.FloatTensor(x)
        x = self.lr(x)
        return x.squeeze()

    def forward(self, x):
        if ~isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = self.lr(x)
        x = 1-1/(1+torch.exp(x))
        return x.squeeze()
    
    def fit(self, x, y, c=1.0, epoch_num=1000, verbose=False, use_sklearn=False):
        if use_sklearn:
            self.sklearn_svc.fit(x, y)
            self.C = self.sklearn_svc.C
            self.lr.weight.data = torch.FloatTensor(self.sklearn_svc.coef_)
            self.lr.bias.data = torch.FloatTensor(self.sklearn_svc.intercept_)
        else:
            criterion = svm_loss_torch
            self.C = c
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y)
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
        return self.forward(torch.FloatTensor(x)).detach().numpy()

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 20)
        self.sm1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(20, 1)
        self.sm2 = torch.nn.Sigmoid()
        self.initialize_weights(self.fc1)
        self.initialize_weights(self.fc2)
        self.input_size = input_size

        # best result according to grid search
        self.sklearn_nn = MLPClassifier(random_state=0, alpha=0.01, learning_rate = 'adaptive', batch_size=1024,\
                                        solver = 'adam', hidden_layer_sizes=(20,), activation='logistic')

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.sm1(x)
        x = self.fc2(x)
        x = self.sm2(x)
        return x.squeeze()
    
    def fit(self, x, y, c=0.005, epoch_num=2000, verbose=False, use_sklearn=False):
        if use_sklearn:
            self.sklearn_nn.fit(x, y)
            self.fc1.weight.data = torch.FloatTensor(self.sklearn_nn.coefs_[0]).T
            self.fc1.bias.data = torch.FloatTensor(self.sklearn_nn.intercepts_[0]).T
            self.fc2.weight.data = torch.FloatTensor(self.sklearn_nn.coefs_[1]).T
            self.fc2.bias.data = torch.FloatTensor(self.sklearn_nn.intercepts_[1]).T
        else:
            self.C = c
            criterion = logistic_loss_torch
            optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y)
            self.train()
            for _ in range(epoch_num):
                loss = criterion(self, x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict_proba(self, x):
        self.eval()
        x = torch.FloatTensor(x).view(-1, self.input_size)
        return self.forward(x).detach().numpy()
    
    def initialize_weights(self, m):
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)
