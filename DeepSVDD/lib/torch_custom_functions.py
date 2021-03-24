"""
pytorch custom function 정의
"""
import torch
import torch.nn

def mse(y_pred, y_true):
    return torch.mean((y_true - y_pred)**2)

def ce(y_pred, y_true):
    return -torch.mean(y_true*(torch.log(y_pred)) + (1-y_true)*(torch.log(1-y_pred)))

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
