import os
import enum
import torch

PRINT_STEP = 1
EXPERIMENT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
SPLIT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'splits')
CHECKPOINT_FORMAT = 'checkpoint_epoch{}.pth.tar'

WEIGHT_INIT = {
    'xavier_normal': torch.nn.init.xavier_normal_,
    'xavier_uniform': torch.nn.init.xavier_uniform_,
    'normal': torch.nn.init.normal_,
    'uniform': torch.nn.init.uniform_,
    'kaiming_normal': torch.nn.init.kaiming_normal_,
    'kaiming_uniform': torch.nn.init.kaiming_uniform_
}

OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam
}

LOSSES = {
    'mse': torch.nn.MSELoss,
    'cross_entropy': torch.nn.CrossEntropyLoss()
}

SCHEDULERS = {
    'step_lr': torch.optim.lr_scheduler.StepLR
}


class SplitType(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'


class InputType(enum.Enum):
    RGB = 'rgb'
    FLOW = 'flow'
    DEPTH = 'depth'
    SKELETON = 'skeleton'


class SamplingType(enum.Enum):
    ALL = 'all'
    RANDOM = 'random'
    EQUIDISTANT = 'equidistant'
    KEYFRAME = 'keyframe'
