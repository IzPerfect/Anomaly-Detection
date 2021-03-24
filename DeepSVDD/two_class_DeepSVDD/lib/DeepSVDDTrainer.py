import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class DeepSVDDTrainer(object):
    def __init__(self, net, device):
        self.net = net

        self.R = torch.tensor(1., device = device)
        self.nu = 0.1
        self.c = torch.zeros(self.net.rep_dim, device=device)

    def init_center_c(self, train_loader, device, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        self.net.eval()

        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = self.net(inputs)
                n_samples += outputs.shape[0]
                self.c += torch.sum(outputs, dim=0)

        self.c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(abs(self.c) < eps) & (self.c > 0)] = eps

        return self.c

    def get_radius(self, dist, nu):
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

    def train(self, class_num, train_loader, device,  epochs = 10, lr = 1e-3, weight_decay = 1e-6, objective = 'soft-boundary', warm_up_n_epochs=10):
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay= weight_decay)

        # Training
        self.net.train() # train mode
        for epoch in range(epochs):
            optimizer.step()

            loss_epoch = 0.0
            n_batches = 0

            for data, Y in train_loader:
                inputs, labels = data, Y
                inputs = inputs[labels==class_num] # normal class settings

                inputs = inputs.to(device)

                optimizer.zero_grad()
                outputs = self.net(inputs)

                dist = torch.sum((outputs - self.c) ** 2, dim=1) # distance from center point

                if objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (objective == 'soft-boundary') and (epoch >= warm_up_n_epochs):
                    self.R = torch.tensor(self.get_radius(dist, self.nu), device=device)

                loss_epoch += loss.item()
                n_batches += 1

            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, loss_epoch / n_batches))


        return self.R, self.c

    def test(self, class_num, test_loader, device, objective = 'soft-boundary'):
        with torch.no_grad():
            self.net.eval() # evalutation mode
            label_score = []

            for data, Y in test_loader:
                inputs, labels = data, Y
                inputs = inputs.to(device)

                outputs = self.net(inputs)

                dist = torch.sum((outputs - self.c) ** 2, dim=1) # distance from center point

                if objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                    scores.cpu().data.numpy().tolist(),
                                    outputs.cpu().data.numpy().tolist()))

        return label_score
