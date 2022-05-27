# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import torch
from bitsandbytes.optim.optimizer import BNBOptimizer

class AdamW(BNBOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=1e-2, amsgrad=False, optim_bits=32, args=None, min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        super(AdamW, self).__init__('adam', params, lr, betas, eps,
                weight_decay, optim_bits, args, min_8bit_size, skip_zeros, quant_maps_or_name)

class AdamW8bit(BNBOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=1e-2, amsgrad=False, args=None, min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        super(AdamW8bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 8, args, min_8bit_size, skip_zeros, quant_maps_or_name)

class AdamW32bit(BNBOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=1e-2, amsgrad=False, args=None, min_8bit_size=204800, skip_zeros=False, quant_maps_or_name='dynamic'):
        super(AdamW32bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 32, args, min_8bit_size, skip_zeros, quant_maps_or_name)
