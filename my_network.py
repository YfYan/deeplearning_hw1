#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:14:59 2022

@author: yanyifan
"""


import numpy as np
import scipy.io as scio
import pickle

np.random.seed(0)
def label2onehot(y:np.ndarray,label_num:int):
    obs_num = y.shape[0]
    y_one_hot = np.zeros(shape = (obs_num,label_num))
    y_one_hot[np.arange(obs_num),y] = 1
    return y_one_hot

def accuracy(y_pred:np.array,y_truth:np.array):
    return (y_pred == y_truth).sum() / y_pred.shape[0]


def data_load():
    mnist = scio.loadmat('mnist-original.mat')
    X = mnist['data'].astype('float64')
    y = mnist['label'].astype('int64').reshape(-1)
    X_train = X[:,:60000].T
    X_test = X[:,60000:].T
    y_train = y[:60000]
    y_test = y[60000:]
    return X_train,y_train,X_test,y_test
    

def save_model(model,name):
    with open(f'./models/{name}.txt','wb') as f:
        pickle.dump(model,f)
        
def load_model(name):
    with open(f'./models/{name}.txt','rb') as f:
        model = pickle.load(f)
    return model

class TwoLayerNetwork(object):
    
    def __init__(self,input_dim, hidden_dim,output_dim,activation = 'relu',reg = 0.001):
        # randomly generate the weight
        self.w1 = np.random.normal(size = (input_dim,hidden_dim)) * 0.1
        self.w2 = np.random.normal(size = (hidden_dim,output_dim)) * 0.1
        self.activation = activation
        self.reg = reg
        
    @staticmethod
    def softmax(x):
        # just avoid infinity number in np.exp()
        x_demax = x - x.max(axis = 1).reshape((-1,1))
        exp_x = np.exp(x_demax)
        prob = exp_x/exp_x.sum(axis = 1).reshape((-1,1))
        return prob
    
    @staticmethod
    def sigmoid_activation(x):
        return 1/(1 + np.exp(-x))
    
    @staticmethod
    def relu_activation(x):
        return np.maximum(x,0)
        
        
    def call(self,x:np.ndarray):
        # forward pass
        self.z1 = x @ self.w1
        
        if self.activation == 'sigmoid':
            self.h = self.sigmoid_activation(self.z1)
        elif self.activation == 'relu':
            self.h = self.relu_activation(self.z1)
        else:
            raise NotImplementedError
        
        self.z2= self.h @ self.w2
        
        self.prob = self.softmax(self.z2)
        
        return self.prob
    
    def loss(self,y_truth:np.ndarray):
        batch_size,label_sum = y_truth.shape
        loss = - np.sum(y_truth[self.prob>1e-6] * np.log(self.prob[self.prob > 1e-6]))/batch_size
        return loss + 0.5*self.reg*np.sum(self.w1 **2 ) + 0.5*self.reg*np.sum(self.w2**2)
    
    def predict(self,x_test:np.ndarray):
        prob = self.call(x_test)
        return prob.argmax(axis = 1)
    
    def backward(self,x:np.ndarray,y_truth:np.ndarray):
        batch_size,label_sum = y_truth.shape
        
        # gradient of self.z2
        grad_z2 = self.prob - y_truth
        
        # gradient of self.w2
        grad_w2 = self.h.T @ grad_z2
        
        # gradient of self.h
        grad_h = grad_z2 @ self.w2.T
        
        # gradient of self.z1
        if self.activation == 'sigmoid':
            grad_z1 = grad_h * (self.h * (1 - self.h))
        elif self.activation == 'relu':
            grad_z1 = grad_h.copy()
            grad_z1[self.h == 0] = 0 
        
        # gradient of self.w1
        grad_w1 = x.T @ grad_z1
        
        return grad_w1/batch_size + self.reg * self.w1 ,grad_w2/batch_size + self.reg * self.w2
    
    def save_parameter(self):
        np.savez('./net_parameter.npz',self.w1,self.w2)
        
    def load_parameter(self,path = './net_parameter.npz'):
        load_data = np.load(path)
        self.w1 = load_data['arr_0']
        self.w2 = load_data['arr_1']
        
        
class AdamOptimizer(object):
    
    def __init__(self,lr):
        self.u1 = None
        self.v1 = None
        
        self.u2 = None
        self.v2 = None
        
        self.lr = lr
        self.mu1 = 0.8
        self.mu2 = 0.8
        
        self.cnt = 0
        
    def update(self,grad_w1,grad_w2):
        eps = 1e-6
        if self.u1 is None:
            self.u1 = grad_w1
            self.v1 = grad_w1 ** 2
            
            self.u2 = grad_w2
            self.v2 = grad_w2 ** 2
        
        else:
            self.u1 = self.mu1 * self.u1 + (1-self.mu1) * grad_w1
            self.v1 = self.mu2 * self.v1 + (1-self.mu2) * grad_w1**2
            
            self.u2 = self.mu1 * self.u2 + (1-self.mu1) * grad_w2
            self.v2 = self.mu2 * self.v2 + (1-self.mu2) * grad_w2**2
            
        delta_w1 = -self.lr/np.sqrt(self.v1 + eps)*self.u1
        delta_w2 = -self.lr/np.sqrt(self.v2 + eps)*self.u2
        
        # self.cnt += 1
        # if self.cnt % 50 == 0:
        #     self.lr /= 2
        return delta_w1,delta_w2
            
        
        
def training(hidden_num,reg,lr,activation = 'relu',verbose = True):
    # using SGD to train the model
    X_train,y_train,X_test,y_test = data_load()

    features,labels = 784,10
    
    y_train_onehot = label2onehot(y_train, labels)
    y_test_onehot = label2onehot(y_test, labels)

    net = TwoLayerNetwork(features, hidden_num, labels,activation,reg)
    optimizer = AdamOptimizer(lr)
    # maximum number of epoch in training
    max_epoch = 500
    # patience for early stopping
    patience = 20
    
    best_itr = 0
    best_val_acc = 0
    
    train_loss_arr = []
    val_loss_arr = []
    
    train_acc_arr = []
    val_acc_arr = []
    
    test_acc_arr = []
    test_loss_arr = []
    for itr in range(max_epoch):
        
        # randomly choose Train and Validation
        idx = np.arange(60000)
        np.random.shuffle(idx)
        
        train_idx = idx[:54000]
        val_idx = idx[54000:]
        
        X = X_train[train_idx]
        y = y_train_onehot[train_idx]
        y_flat = y_train[train_idx]
        
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        y_val_onehot = y_train_onehot[val_idx]
        
        
        _ = net.call(X)
        
        grad_w1,grad_w2 = net.backward(X,y)
        # self-adaptive gradient descent
        delta_w1,delta_w2 = optimizer.update(grad_w1,grad_w2)
        
        
        loss = net.loss(y)
        pred = net.predict(X)
        train_acc = accuracy(pred,y_flat)
        if verbose:
            print("train acc:",train_acc)
            print("train loss:",loss)
        train_acc_arr.append(train_acc)
        train_loss_arr.append(loss)
        
        y_pred = net.predict(X_val)
        val_loss = net.loss(y_val_onehot)
        val_acc = accuracy(y_pred,y_val)
        if verbose:
            print("val_accuracy:",val_acc)
            print("val_loss,",val_loss)
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)
        
        y_pred_test = net.predict(X_test)
        test_loss = net.loss(y_test_onehot)
        test_acc = accuracy(y_pred_test,y_test)
        
        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_acc)
        
        
        if best_val_acc < val_acc:
            best_val_acc  = val_acc
            best_itr = itr
        
        if itr -  best_itr > patience:
            break
        
        net.w1 += delta_w1
        net.w2 += delta_w2
        
        
    
    history = {}
    history['train loss'] = train_loss_arr
    history['train acc'] = train_acc_arr
    history['val loss'] = val_loss_arr
    history['val acc'] = val_acc_arr
    history['test loss'] = test_loss_arr
    history['test acc'] = test_acc_arr
    return net,history

        
    
if __name__ == '__main__':
    training(100,0.001,0.001)
      
    
    
    
        
        
        
        
        
    