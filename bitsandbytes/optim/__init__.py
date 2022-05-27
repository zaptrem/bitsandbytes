# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
from .adam import Adam, Adam8bit, Adam32bit, Adam0bit
from .adamw import AdamW, AdamW8bit, AdamW32bit
from .sgd import SGD, SGD8bit, SGD32bit
from .rmsprop import RMSprop, RMSprop8bit, RMSprop32bit
from .adagrad import Adagrad, Adagrad8bit, Adagrad32bit
from .optimizer import GlobalOptimManager
