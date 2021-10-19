# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import pytest
import os
import math
import json
import torch
import bitsandbytes as bnb

from itertools import product

from bitsandbytes.optim import Adam, LAMB, LARS, SGD, RMSprop


''' These regression tests are meant to test that optimizers do
    not change in any meaningful way. The current implementations
    are correct and any change in outputs is captured by these
    tests and needs to be justified.
'''
name2optimizer = {}
name2optimizer['Adam'] = Adam
name2optimizer['LAMB'] = LAMB
name2optimizer['LARS'] = lambda p, optim_bits: LARS(p, optim_bits=optim_bits, lr=0.01, momentum=0.9)
name2optimizer['SGD'] = lambda p, optim_bits: SGD(p, optim_bits=optim_bits, lr=0.001, momentum=0.9)
name2optimizer['RMSprop'] = lambda p, optim_bits: RMSprop(p, optim_bits=optim_bits, lr=0.001)

# only set this if there is a good reason to overwrite regression data
OVERWRITE_REGRESSION_DATA = False
overwrite_keys = ['Adam', 'LAMB', 'LARS', 'SGD', 'RMSprop'] # only overwrite specific optimizers
#overwrite_keys = ['LAMB', 'LARS']

@pytest.fixture
def regression_data():
    if not os.path.exists('./tests/regression_data.json'):
        data = {}
        for name in name2optimizer:
            data[name] = {}
        with open('./tests/regression_data.json', 'w') as f:
            json.dump(data, f)
    else:
        with open('./tests/regression_data.json') as f:
            data = json.load(f)

    return data


def save_reression_data(data):
    with open('./tests/regression_data.json', 'w') as f:
        json.dump(data, f)

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
dim1 = [2048, 3072, 1583]
dim2 = [2048, 3072, 1283]
#dim1 = [2048]
#dim2 = [2048]
gtype = [torch.float32, torch.float16]
#gtype = [torch.float32]
optimizers_names = ['Adam', 'LAMB', 'LARS', 'SGD', 'RMSprop']
optim_bits = [8, 32]
#optim_bits = [32]
values = list(product(dim1,dim2, gtype, optim_bits, optimizers_names))
values_names = list(product(dim1,dim2, gtype, optim_bits, optimizers_names))
names = ['dim1_{0}_dim2_{1}_gtype_{2}_bits_{3}_optim_{4}'.format(*vals) for vals in values_names]
@pytest.mark.parametrize("dim1, dim2, gtype, optim_bits, optim_name", values, ids=names)
def test_regression_optimizers(regression_data, dim1, dim2, gtype, optim_bits, optim_name):
    optim_cls = name2optimizer[optim_name]
    testid = f'{dim1}_{dim2}_{gtype}_{optim_bits}_{str(optim_name)}'
    print(testid)
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

    epoch_losses = []
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
        epoch_losses.append(sum(losses)/len(losses))

    if OVERWRITE_REGRESSION_DATA and optim_name in overwrite_keys:
        regression_data[optim_name][testid] = epoch_losses
        save_reression_data(regression_data)
    else:
        expected = regression_data[optim_name][testid]
        expected = torch.tensor(expected)
        actual = torch.tensor(epoch_losses)
        torch.testing.assert_allclose(actual, expected)









