# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
from bitsandbytes.optim.optimizer import BNBOptimizer

class SGD(BNBOptimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, optim_bits=32, args=None,
            min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        if momentum == 0:
            raise NotImplementedError(f'SGD without momentum is not supported!')
        super(SGD, self).__init__('momentum', params, lr, (momentum, dampening), 0.0,
                weight_decay, optim_bits, args, min_8bit_size, skip_zeros, quant_maps_or_name)

class SGD8bit(BNBOptimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, args=None,
                 min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        if momentum == 0:
            raise NotImplementedError(f'SGD without momentum is not supported!')
        super(SGD8bit, self).__init__('momentum', params, lr, (momentum, dampening), 0.0,
                weight_decay, 8, args, min_8bit_size, skip_zeros, quant_maps_or_name)

class SGD32bit(BNBOptimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, args=None,
                 min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        if momentum == 0:
            raise NotImplementedError(f'SGD without momentum is not supported!')
        super(SGD32bit, self).__init__('momentum', params, lr, (momentum, dampening), 0.0,
                weight_decay, 32, args, min_8bit_size, skip_zeros, quant_maps_or_name)


