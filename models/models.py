import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_

from layers import *


# def Encoder(nn.Module):
#     def __init__(self, dim, )