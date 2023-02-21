import os
import sys
import torch
import random
import numpy as np
import importlib
from torch.optim.lr_scheduler import _LRScheduler


def set_random_seed(seed, deterministic, benchmark):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:           # 固定内部随机性
        torch.backends.cudnn.deterministic = True
    if benchmark:               # 输入尺寸一致时，加速训练
        torch.backends.cudnn.benchmark = True


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


'''
读取配置文件
'''
def file2dict(filename):
    (path,file) = os.path.split(filename)

    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0,abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
                and not isinstance(value, types.ModuleType)
                and not isinstance(value, types.FunctionType)
                    }
    return cfg_dict.get('train_pipeline'),cfg_dict.get('val_pipeline'),cfg_dict.get('data_cfg'),cfg_dict.get('lr_config'),cfg_dict.get('optimizer_cfg')