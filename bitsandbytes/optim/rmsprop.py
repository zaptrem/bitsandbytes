# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import torch
from bitsandbytes.optim.optimizer import BNBOptimizer

class RMSprop(BNBOptimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, optim_bits=32, args=None,
            min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        if alpha == 0:
            raise NotImplementedError(f'RMSprop with alpha==0.0 is not supported!')
        if centered:
            raise NotImplementedError(f'Centered RMSprop is not supported!')
        super(RMSprop, self).__init__('rmsprop', params, lr, (alpha, momentum), eps,
                weight_decay, optim_bits, args, min_8bit_size, skip_zeros, quant_maps_or_name)

class RMSprop8bit(BNBOptimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, args=None,
            min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        if alpha == 0:
            raise NotImplementedError(f'RMSprop with alpha==0.0 is not supported!')
        if centered:
            raise NotImplementedError(f'Centered RMSprop is not supported!')
        super(RMSprop8bit, self).__init__('rmsprop', params, lr, (alpha, momentum), eps,
                weight_decay, 8, args, min_8bit_size, skip_zeros, quant_maps_or_name)

class RMSprop32bit(BNBOptimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, args=None,
            min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):

        if alpha == 0:
            raise NotImplementedError(f'RMSprop with alpha==0.0 is not supported!')
        if centered:
            raise NotImplementedError(f'Centered RMSprop is not supported!')
        super(RMSprop32bit, self).__init__('rmsprop', params, lr, (alpha, momentum), eps,
                weight_decay, 32, args, min_8bit_size, skip_zeros, quant_maps_or_name)
