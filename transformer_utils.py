import torch
import torch.nn as nn

import copy

def clone_modules(module, N):
    # generates N identical torch modules
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




