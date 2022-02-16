import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


# inputs:
#   - nets (int): number of networks to create
#   - nodes (list): each element corresponds to the number of number of neurons
#                   in each layer - need to specify input neuron count as well
#   - act_f (list): each element corresponds to its index's layer's activation
#                   function
#   - opt (torch.optim.OPT): PyTorch optimizer
#   - loss_fn (function pointer): reference to the desired loss function
#   - lr (float) [optional]: learning rate
#   - device (torch.device) [optional]: tries to train using gpu by default
class MultiNetwork(nn.Module):
    def __init__(self, nets, nodes, act_f, opt, loss_fn, lr=0.001, device=None):
        super(MultiNetwork, self).__init__()
        self.nets = nets
        self.layers = []
        for n in range(self.nets):
            self.layers.append([])
            for l in range(len(nodes)-1):
                self.layers[n].append(nn.Linear(nodes[l],nodes[l+1]))
        self.out_len = nodes[-1]
        self.act_f = act_f
        self.opt = opt(self.parameters(),lr=lr)
        self.loss_fn = loss_fn
        self.device = device
        if self.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parameters(self):
        p = []
        for n in range(self.nets):
            for l in self.layers[n]:
                p.append(l.weight)
                p.append(l.bias)
        return p

    def parameters_i(self, i):
        p = []
        for l in self.layers[i]:
            p.append(l.weight)
            p.append(l.bias)
        return p

    def forward(self, x):
        y = torch.empty((x.shape[0],self.nets*self.out_len)).to(self.device)
        for n in range(self.nets):
            out = x.to(self.device)
            for i, l in enumerate(self.layers[n]):
                out = self.act_f[i](l.to(self.device)(out))
            y[:,n*self.out_len:(n+1)*self.out_len] = out
        return y

    def loss(self, y_hat, y):
        loss = 0.0
        losses = []
        for n in range(self.nets):
            curr_loss = self.loss_fn(y_hat[:,n*self.out_len:(n+1)*self.out_len],y.to(self.device))
            loss += curr_loss
            losses.append(curr_loss.item())
        return loss, losses

    def train(self, x, y):
        self.opt.zero_grad()
        loss, ls = self.loss(self.forward(x), y)
        loss.backward()
        self.opt.step()
        return loss.item(), ls

    def get_network(self, index):
        return self.layers[index]
