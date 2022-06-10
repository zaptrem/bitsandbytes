# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import os
import time
import shutil
import uuid
import pytest
import ctypes
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

from os.path import join
from itertools import product

#import apex

def get_temp_dir():
    path = '/tmp/autoswap/{0}'.format(str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path

def rm_path(path):
    shutil.rmtree(path)

str2optimizers = {}
str2optimizers['adam_pytorch'] = (None, torch.optim.Adam, bnb.optim.Adam)
#str2optimizers['adam_apex'] = (None, apex.optimizers.FusedAdam, bnb.optim.Adam)
#str2optimizers['momentum_apex'] = (None, lambda pxx: apex.optimizers.FusedSGD(pxx, 0.01, 0.9), bnb.optim.Adam)
str2optimizers['momentum_pytorch'] = (None, lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9), bnb.optim.Adam)

str2optimizers['adam'] = (torch.optim.Adam, bnb.optim.Adam)
str2optimizers['adamw'] = (torch.optim.AdamW, bnb.optim.AdamW)
#str2optimizers['fused_adam'] = (apex.optimizers.FusedAdam, bnb.optim.Adam)
str2optimizers['momentum'] = (lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9), lambda pxx: bnb.optim.SGD(pxx, 0.01, 0.9))
str2optimizers['rmsprop'] = (lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9), lambda pxx: bnb.optim.RMSprop(pxx, 0.01, 0.9))
str2optimizers['adagrad'] = (lambda pxx: torch.optim.Adagrad(pxx, 0.01), lambda pxx: bnb.optim.Adagrad(pxx, 0.01))

str2optimizers['adam8bit_blockwise'] = (torch.optim.Adam, bnb.optim.Adam8bit)
str2optimizers['adamw8bit_blockwise'] = (torch.optim.Adam, lambda pxx: bnb.optim.AdamW8bit(pxx))
str2optimizers['momentum8bit_blockwise'] = (lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9), lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9))
str2optimizers['rmsprop8bit_blockwise'] = (lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9), lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9))
str2optimizers['adagrad8bit_blockwise'] = (lambda pxx: torch.optim.Adagrad(pxx, 0.01), lambda pxx: bnb.optim.Adagrad8bit(pxx, 0.01))

str2optimizers['adam8bit_streaming'] = (torch.optim.Adam, bnb.optim.Adam0bit)

str2statenames = {}
str2statenames['adam'] = [('exp_avg', 'state1'), ('exp_avg_sq', 'state2')]
str2statenames['adamw'] = [('exp_avg', 'state1'), ('exp_avg_sq', 'state2')]
str2statenames['momentum'] = [('momentum_buffer', 'state1')]
str2statenames['rmsprop'] = [('square_avg', 'state1')]
str2statenames['adagrad'] = [('sum', 'state1')]
str2statenames['adam8bit_blockwise'] = [('exp_avg', 'state1', 'qmap1', 'absmax1'), ('exp_avg_sq', 'state2', 'qmap2', 'absmax2')]
str2statenames['adam8bit_streaming'] = [('exp_avg', 'state1', 'qmap1', 'absmax1'), ('exp_avg_sq', 'state2', 'qmap2', 'absmax2')]
str2statenames['adamw8bit_blockwise'] = [('exp_avg', 'state1', 'qmap1', 'absmax1'), ('exp_avg_sq', 'state2', 'qmap2', 'absmax2')]
str2statenames['momentum8bit_blockwise'] = [('momentum_buffer', 'state1', 'qmap1', 'absmax1')]
str2statenames['rmsprop8bit_blockwise'] = [('square_avg', 'state1', 'qmap1', 'absmax1')]
str2statenames['adagrad8bit_blockwise'] = [('sum', 'state1', 'qmap1', 'absmax1')]

dim1 = [1024]
dim2 = [32, 1024, 4097, 1]
gtype = [torch.float32, torch.float16]
optimizer_names = ['adam', 'adamw', 'momentum', 'rmsprop', 'adagrad']
values = list(product(dim1,dim2, gtype, optimizer_names))
names = ['dim1_{0}_dim2_{1}_gtype_{2}_optim_{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name", values, ids=names)
def test_optimizer32bit(dim1, dim2, gtype, optim_name):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1
    p2 = p1.clone()
    p1 = p1.float()


    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 2e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3


    for i in range(50):
        g = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()

        for name1, name2 in str2statenames[optim_name]:
            torch.testing.assert_allclose(torch_optimizer.state[p1][name1], bnb_optimizer.state[p2][name2], atol=atol, rtol=rtol)

        torch.testing.assert_allclose(p1, p2.float(), atol=atol, rtol=rtol)

        if i % 10 == 0 and i > 0:
            path = get_temp_dir()
            torch.save(bnb_optimizer.state_dict(),join(path, 'opt.pt'))
            del bnb_optimizer
            bnb_optimizer = None
            bnb_optimizer = str2optimizers[optim_name][1]([p2])
            bnb_optimizer.load_state_dict(torch.load(join(path, 'opt.pt')))
            rm_path(path)
            torch.testing.assert_allclose(p1, p2.float(), atol=atol, rtol=rtol)
            for name1, name2 in str2statenames[optim_name]:
                torch.testing.assert_allclose(torch_optimizer.state[p1][name1], bnb_optimizer.state[p2][name2], atol=atol, rtol=rtol)

        if gtype == torch.float16:
            # the adam buffers should also be close because they are 32-bit
            # but the paramters can diverge because they are 16-bit
            # the difference grow larger and larger with each update
            # --> copy the state to keep weights close
            p1.data = p1.data.half().float()
            p2.copy_(p1.data)
            torch.testing.assert_allclose(p1.half(), p2)

dim1 = [1024]
dim2 = [1024, 4097]
gtype = [torch.float32, torch.float16]
values = list(product(dim1,dim2, gtype))
names = ['dim1_{0}_dim2_{1}_gtype_{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype", values, ids=names)
def test_global_config(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.nn.Linear(dim1,dim2).to(gtype)
    p2 = torch.nn.Linear(dim1,dim2).to(gtype)
    p3 = torch.nn.Linear(dim1,dim2).to(gtype)
    mask = torch.rand_like(p2.weight) < 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8

    bnb.optim.GlobalOptimManager.get_instance().initialize()
    bnb.optim.GlobalOptimManager.get_instance().register_module_override(p2, 'weight', {'skip_zeros': True})
    bnb.optim.GlobalOptimManager.get_instance().register_module_override(p3, 'weight', {'optim_bits': 8})



    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()

    adam2 = bnb.optim.Adam([p1.weight, p2.weight, p3.weight], lr, (beta1, beta2), eps)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3

    original_p2 = p2.weight[mask].clone()

    for i in range(50):
        g1 = torch.randn(dim2,dim1, device='cuda', dtype=gtype)*0.1 + 0.001
        g2 = torch.randn(dim2,dim1, device='cuda', dtype=gtype)*0.1 + 0.001
        g3 = torch.randn(dim2,dim1, device='cuda', dtype=gtype)*0.1 + 0.001
        p1.weight.grad = g1
        p2.weight.grad = g2
        p3.weight.grad = g3

        if i > 30 and i % 10 == 0:
            g1.data[mask] = 0.0
            g2.data[mask] = 0.0
            p1.weight.grad = g1
            p2.weight.grad = g2
            original_p1 = p1.weight[mask].clone()
            original_p2 = p2.weight[mask].clone()
            og_s1 = adam2.state[p2.weight]['state1'][mask].clone()
            og_s2 = adam2.state[p2.weight]['state2'][mask].clone()
            og_s11 = adam2.state[p1.weight]['state1'][mask].clone()
            og_s21 = adam2.state[p1.weight]['state2'][mask].clone()

        adam2.step()

        assert adam2.state[p3.weight]['state1'].dtype == torch.uint8
        assert adam2.state[p3.weight]['state2'].dtype == torch.uint8

        if i > 30 and i % 10 == 0:
            torch.testing.assert_allclose(original_p2, p2.weight[mask])
            torch.testing.assert_allclose(adam2.state[p2.weight]['state1'][mask], og_s1)
            torch.testing.assert_allclose(adam2.state[p2.weight]['state2'][mask], og_s2)
            assert ((p1.weight[mask]- original_p1)==0.0).sum() < p1.weight.numel()
            assert ((adam2.state[p1.weight]['state1'][mask]- og_s11)==0.0).sum() == 0.0
            assert ((adam2.state[p1.weight]['state2'][mask]- og_s21)==0.0).sum() == 0.0




dim1 = [1024]
dim2 = [1024, 4097]
gtype = [torch.float32, torch.float16]
optimizer_names = ['adam8bit_blockwise', 'adamw8bit_blockwise', 'momentum8bit_blockwise', 'rmsprop8bit_blockwise', 'adagrad8bit_blockwise', 'adam8bit_streaming']
values = list(product(dim1,dim2, gtype, optimizer_names))
names = ['dim1_{0}_dim2_{1}_gtype_{2}_optim_{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name", values, ids=names)
def test_optimizer8bit(dim1, dim2, gtype, optim_name):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1
    p2 = p1.clone()
    p1 = p1.clone().float()
    blocksize = 2048

    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    errors = []
    relerrors = []

    for i in range(50):
        g = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()

        torch.testing.assert_allclose(p1, p2.float(), atol=patol, rtol=prtol)

        dequant_states = []
        for name1, name2, qmap, max_val in str2statenames[optim_name]:
            #print(bnb_optimizer.state[p2][max_val], name1)
            #if 'blockwise' in optim_name:
            #    # absmax none
            s1 = F.dequantize(absmax=bnb_optimizer.state[p2][max_val], A=bnb_optimizer.state[p2][name2], blocksize=blocksize, dtype=torch.float32, is_signed=name2=='state1')
            s32 = torch_optimizer.state[p1][name1]
            idx = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol)==0

            num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol)==0
            assert num_not_close.sum().item() < 20
            dequant_states.append(s1.clone())

        err  = torch.abs(p1-p2)
        relerr = err/torch.abs(p1)
        assert err.mean() < 0.0001
        assert relerr.mean() < 0.001

        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())

        if i % 10 == 0 and i > 0:
            for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
                s1cpy = s.clone()
                raws1cpy = bnb_optimizer.state[p2][name2].clone().cuda()
                #qmap1 = bnb_optimizer.state[p2][qmap].clone()

                path = get_temp_dir()
                torch.save(bnb_optimizer.state_dict(),join(path, 'opt.pt'))
                del bnb_optimizer
                bnb_optimizer = None
                bnb_optimizer = str2optimizers[optim_name][1]([p2])
                bnb_optimizer.load_state_dict(torch.load(join(path, 'opt.pt')))
                rm_path(path)
                torch.testing.assert_allclose(raws1cpy, bnb_optimizer.state[p2][name2])
                #torch.testing.assert_allclose(qmap1, bnb_optimizer.state[p2][qmap])

                s1 = F.dequantize(absmax=bnb_optimizer.state[p2][max_val], A=bnb_optimizer.state[p2][name2], blocksize=blocksize, dtype=torch.float32, is_signed=name2=='state1')
                torch.testing.assert_allclose(s1cpy, s1)

                num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol)==0
                assert num_not_close.sum().item() < 20
            torch.testing.assert_allclose(p1, p2.float(), atol=patol, rtol=prtol)

        # the parameters diverge quickly. Here we keep them close
        # together so we can test against the Adam error
        p1.data = p1.data.to(gtype).float()
        p2.copy_(p1.data)
        torch.testing.assert_allclose(p1.to(gtype), p2)
        for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
            torch_optimizer.state[p1][name1].copy_(s.data)

    #print(sum(errors)/len(errors))
    #print(sum(relerrors)/len(relerrors))



dim1 = [4096]
dim2 = [4096]
gtype = [torch.float32, torch.float16]
#optimizer_names = ['adam8bit_blockwise', 'adam8bit', 'lamb8bit']
#optimizer_names = ['adam8bit_blockwise', 'adam_apex', 'adam8bit', 'adam', 'adam_pytorch']
#optimizer_names = ['momentum_apex', 'momentum8bit', 'momentum_pytorch']
optimizer_names = ['adam8bit_blockwise']
values = list(product(dim1,dim2, gtype, optimizer_names))
names = ['dim1_{0}_dim2_{1}_gtype_{2}_optim_{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name", values, ids=names)
def test_benchmark_blockwise(dim1, dim2, gtype, optim_name):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1


    bnb_optimizer = str2optimizers[optim_name][1]([p1])

    g = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.01
    p1.grad = g
    for i in range(5000):
        if i == 500:
            # 100 iterations for burn-in
            torch.cuda.synchronize()
            t0 = time.time()

        bnb_optimizer.step()

    torch.cuda.synchronize()
    s = time.time()-t0
    print('')
    params = 4500*4096*4096
    print(optim_name, gtype, s/params)
    #assert s < 3.9



def test_str_betas():
    betas = (0.80, 0.95)
    strbetas = '(0.80, 0.95)'

    layer = torch.nn.Linear(10, 10)

    base = bnb.optim.Adam(layer.parameters(), betas=betas)
    strbase = bnb.optim.Adam(layer.parameters(), betas=strbetas)
    assert base.defaults['betas'][0] == 0.8
    assert base.defaults['betas'][1] == 0.95
    assert strbase.defaults['betas'][0] == 0.8
    assert strbase.defaults['betas'][1] == 0.95




dim1 = [18*1024]
gtype = [torch.float16]
optimizer_names = ['adam8bit_streaming']
values = list(product(dim1,gtype, optimizer_names))
names = ['dim1_{0}_gtype_{1}_optim_{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, gtype, optim_name", values, ids=names)
def test_stream_optimizer_bench(dim1, gtype, optim_name):
    layers1 = torch.nn.Sequential(*torch.nn.ModuleList([torch.nn.Linear(dim1, dim1) for i in range(10)]))
    layers1 = layers1.to(gtype)
    layers1 = layers1.cuda()

    #torch_optimizer = str2optimizers[optim_name][0](layers1.parameters())
    bnb_optimizer = str2optimizers[optim_name][1](layers1.parameters())

    batches = torch.randn(50, 128, dim1, device='cuda').to(gtype)
    lbls = torch.randint(0, 10, size=(50,128)).cuda()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(12):
        b = batches[i]

        out1 = layers1(b)

        loss1 = torch.nn.functional.cross_entropy(out1, lbls[i]).mean()
        loss1.backward()
        #torch_optimizer.step()
        bnb_optimizer.step()
    torch.cuda.synchronize()
    print('pytorch', time.time() - t0)


def get_errors(a, b, k=5):
    err = torch.abs(a-b)
    relerr = err/torch.abs(b)
    errtopk, idx1 = torch.topk(err.flatten(), k=k)
    rellerrtopk, idx2 = torch.topk(relerr.flatten(), k=k)
    return errtopk, rellerrtopk, idx1, idx2

dim1 = [1024]
dim2 = [1024]
gtype = [torch.float32, torch.float16]
values = list(product(dim1,dim2, gtype))
names = ['dim1_{0}_dim2_{1}_gtype_{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype", values, ids=names)
def test_optimizer_bitwise_vs_code(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1: return
    torch.manual_seed(22)
    p1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1
    p2 = p1.clone()
    p1 = p1.clone()
    blocksize = 2048

    code1 = F.create_dynamic_map(True).cuda()
    code2 = F.create_dynamic_map(False).cuda()

    opt1 = bnb.optim.Adam8bit([p1])
    opt2 = bnb.optim.Adam8bit([p2], quant_maps_or_name=[code1, code2])


    print('')
    for i in range(5):
        g = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.01
        p1.grad = g.clone()
        p2.grad = g.clone()

        opt1.step()
        opt2.step()

        err, relerr, erridx, relerridx = get_errors(p1, p2, k=50)
        print(opt1.state[p1]['state1'].flatten()[132407])
        print(opt2.state[p2]['state1'].flatten()[132407]-128)
        print(opt1.state[p1]['state2'].flatten()[132407])
        print(opt2.state[p2]['state2'].flatten()[132407])
        print(p1.flatten()[132407])
        print(p2.flatten()[132407])
        #print(p1)
        #print(p2)
        print(err)
        print(erridx)


