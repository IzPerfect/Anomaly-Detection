import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

# random.seed(777)
# torch.manual_seed(777)


class MyMNIST(Dataset):
  def __init__(self, batch_size = 128):

    self.mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    self.mnist_test = dsets.MNIST(root='MNIST_data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

    self.train_data_loader = torch.utils.data.DataLoader(dataset=self.mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

    self.test_data_loader = torch.utils.data.DataLoader(dataset=self.mnist_test,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
