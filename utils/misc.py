import numpy as np
import torch
import random

        
        
def reproduc(seed, benchmark=False, deterministic=True):
    """Make experiments reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
