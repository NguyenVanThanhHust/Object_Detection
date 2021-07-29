import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Feature stride for RPN
__C.FEAT_STRIDE = [16, ]