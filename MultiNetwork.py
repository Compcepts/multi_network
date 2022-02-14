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
        super(Network, self).__init__()
        self.nets = nets
        self.layers = []
        for l in range(len(nodes)-1):
            if l == 0:
                self.layers.append(nn.Linear(nodes[l],nodes[l+1]*nets))
            else:
                self.layers.append(nn.Linear(nodes[l]*nets,nodes[l+1]*nets))
        self.act_f = act_f
        self.opt = opt(self.parameters(),lr=lr)
        self.loss_fn = loss_fn
        if device != None:
            self.device = device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parameters(self):
        p = []
        for l in self.layers:
            p.append(l.weight)
            p.append(l.bias)
        return p

    def forward(self, x):
        out = x.to(self.device)
        for i, l in enumerate(self.layers):
            out = self.act_f[i](l.to(self.device)(out))
        return out

    def loss(self, y_hat, y):
        loss = 0.0
        losses = []
        out_size = self.layers[-1].weight.shape[1]/self.nets
        for m in range(self.nets):
            curr_loss = self.loss_fn(y_hat[:,(m-1)*out_size:m*out_size],y.to(device))
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
        layers = []
        for i, l in enumerate(self.layers):
            w_shape = [0]*2
            w_shape[0] = self.layers[i].weight.shape[0]
            w_shape[1] = self.layers[i].weight.shape[1]
            w_shape[0] = w_shape[0]/self.nets
            if i != 0:
                w_shape[1] = w_shape[1]/self.nets
            layer = nn.Linear(w_shape[1], w_shape[0])
            layer.weight = self.layers[i].weight[(index-1)*w_shape[0]:index*w_shape[0],:]
            layer.bias = self.layers[i].bias[(index-1)*w_shape[0]:index*w_shape[0]]
            layers.append(layer)
        return layers
