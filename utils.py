import logging
import os
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots"):
        os.makedirs("./snapshots")
    filename = os.path.join("./snapshots/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)

def random_choice(num_choice, layers):
    return list(np.random.randint(num_choice, size=layers))

def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # cudnn.enabled = True
    # cudnn.benchmark = False
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))
