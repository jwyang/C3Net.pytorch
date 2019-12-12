import os
from yacs.config import CfgNode as CN

_C = CN()

_C.data = CN()
_C.data.input_size = 32
_C.data.traindir = ""
_C.data.valdir = ""
_C.data.testdir = ""

_C.initialize = CN()
_C.initialize.seed = 1

_C.training = CN()
_C.training.batch_size = 128
_C.training.workers = 8
_C.training.distributed = False

_C.test = CN()
_C.test.batch_size = 1000
_C.test.workers = 8

_C.optimizer = CN()
_C.optimizer.name = "sgd"
_C.optimizer.lr = 0.1
_C.optimizer.lr_decay_gamma = 0.1
_C.optimizer.weight_decay = 1e-4
_C.optimizer.momentum = 0.9
_C.optimizer.lr_decay_schedule = (60, 120)
_C.optimizer.max_epoch = 160

_C.log = CN()
_C.log.print_interval = 20
_C.log.checkpoint_interval = 2
