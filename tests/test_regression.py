# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import pytest
import math
import torch
import bitsandbytes as bnb

from itertools import product

from bitsandbytes.optim import Adam, LAMB, LARS, SGD, RMSprop


''' These regression tests are meant to test that optimizers do
    not change in any meaningful way. The current implementations
    are correct and any change in outputs is captured by these
    tests and needs to be justified.
'''

class MLP(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(2, dim1)
        self.fc2 = torch.nn.Linear(dim1, dim2)
        self.fc3 = torch.nn.Linear(dim2, dim1)
        self.fc4 = torch.nn.Linear(dim1, 2)
        self.norm1 = torch.nn.LayerNorm(dim1)
        self.norm2 = torch.nn.LayerNorm(dim2)
        self.norm3 = torch.nn.LayerNorm(dim1)

    def forward(self, x):
        x = self.norm1(torch.relu(self.fc1(x)))
        x = self.norm2(torch.relu(self.fc2(x)))
        x = self.norm3(torch.relu(self.fc3(x)))
        x = self.fc4(x)

        return x
        #return torch.nn.functional.hardtanh(x)



# test a common block-wise dimension, i.e. 2048, a non-multiple of that
# 3072, and a value that conflicts which thread-block size, say,
# 1583 (a nice prime number) and 1283 (another nice prime number)
#dim1 = [2048, 3072, 1583]
#dim2 = [2048, 3072, 1283]
dim1 = [2048]
dim2 = [2048]
#gtype = [torch.float32, torch.float16]
gtype = [torch.float32]
optimizers = [Adam, LAMB, lambda p, optim_bits: LARS(p, optim_bits=optim_bits, lr=0.01, momentum=0.9), lambda p, optim_bits: SGD(p,lr=0.001, optim_bits=optim_bits, momentum=0.9), lambda p, optim_bits: RMSprop(p, optim_bits=optim_bits, lr=0.001)]
optimizers_names = ['Adam', 'LAMB', 'LARS', 'SGD', 'RMSprop']
#optim_bits = [8, 32]
optim_bits = [32]
values = list(product(dim1,dim2, gtype, optim_bits, optimizers))
values_names = list(product(dim1,dim2, gtype, optim_bits, optimizers_names))
names = ['dim1_{0}_dim2_{1}_gtype_{2}_bits_{3}_optim_{4}'.format(*vals) for vals in values_names]
@pytest.mark.parametrize("dim1, dim2, gtype, optim_bits, optim_cls", values, ids=names)
def test_regression_optimizers(dim1, dim2, gtype, optim_bits, optim_cls):
    torch.manual_seed(78787887)
    batch_size = 32

    # create some data with signal with structure
    x = torch.cat([torch.linspace(-1.0, 1.0, 1024*batch_size).view(-1, 1), torch.linspace(1, -1, 1024*batch_size).view(-1, 1)], dim=1)
    y = torch.cat([(x[:, 0]*x[:, 1]).view(-1, 1), (x[:, 0]*x[:, 1]*x[:, 0]).view(-1, 1)], dim=1)
    # inputs -> outputs
    # x1*x2 -> [x1*x2, x1*x2*x1]
    assert x[17, 0]*x[17, 1] == y[17, 0]

    x -= x.mean(0)
    x /= x.std(0)

    #y -= y.mean(0)
    #y /= y.std(0)

    idx = torch.randperm(x.shape[0])

    x = x[idx]
    y = y[idx]

    mlp = MLP(dim1, dim2).cuda()
    if gtype == torch.half:
        mlp = mlp.half()

    optimizer = optim_cls(mlp.parameters(), optim_bits=optim_bits)

    for epoch in range(10):
        losses = []
        for i in range(0, 1024, batch_size):
            inputs = x[i:i+batch_size].cuda()
            lbls = y[i:i+batch_size].cuda()

            if gtype == torch.half:
                inputs = inputs.half()
                lbls = lbls.half()


            pred = mlp(inputs)
            loss = torch.nn.functional.mse_loss(pred, lbls)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(epoch, sum(losses)/len(losses))







