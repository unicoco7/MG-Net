# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Rainbowsecret (yuyua@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# configs for HRNet64
HRNET_64 = CN()
HRNET_64.STEM_INPLANES = 64
HRNET_64.FINAL_CONV_KERNEL = 1
HRNET_64.WITH_HEAD = True

HRNET_64.STAGE2 = CN()
HRNET_64.STAGE2.NUM_MODULES = 1
HRNET_64.STAGE2.NUM_BRANCHES = 2
HRNET_64.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_64.STAGE2.NUM_CHANNELS = [64, 128]
HRNET_64.STAGE2.BLOCK = 'BASIC'
HRNET_64.STAGE2.FUSE_METHOD = 'SUM'

HRNET_64.STAGE3 = CN()
HRNET_64.STAGE3.NUM_MODULES = 4
HRNET_64.STAGE3.NUM_BRANCHES = 3
HRNET_64.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_64.STAGE3.NUM_CHANNELS = [64, 128, 256]
HRNET_64.STAGE3.BLOCK = 'BASIC'
HRNET_64.STAGE3.FUSE_METHOD = 'SUM'

HRNET_64.STAGE4 = CN()
HRNET_64.STAGE4.NUM_MODULES = 3
HRNET_64.STAGE4.NUM_BRANCHES = 4
HRNET_64.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_64.STAGE4.NUM_CHANNELS = [64, 128, 256, 512]
HRNET_64.STAGE4.BLOCK = 'BASIC'
HRNET_64.STAGE4.FUSE_METHOD = 'SUM'


# configs for HRNet48
HRNET_48 = CN()
HRNET_48.STEM_INPLANES = 64
HRNET_48.FINAL_CONV_KERNEL = 1
HRNET_48.WITH_HEAD = True

HRNET_48.STAGE2 = CN()
HRNET_48.STAGE2.NUM_MODULES = 1
HRNET_48.STAGE2.NUM_BRANCHES = 2
HRNET_48.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_48.STAGE2.NUM_CHANNELS = [48, 96]
HRNET_48.STAGE2.BLOCK = 'BASIC'
HRNET_48.STAGE2.FUSE_METHOD = 'SUM'

HRNET_48.STAGE3 = CN()
HRNET_48.STAGE3.NUM_MODULES = 4
HRNET_48.STAGE3.NUM_BRANCHES = 3
HRNET_48.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_48.STAGE3.NUM_CHANNELS = [48, 96, 192]
HRNET_48.STAGE3.BLOCK = 'BASIC'
HRNET_48.STAGE3.FUSE_METHOD = 'SUM'

HRNET_48.STAGE4 = CN()
HRNET_48.STAGE4.NUM_MODULES = 3
HRNET_48.STAGE4.NUM_BRANCHES = 4
HRNET_48.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_48.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
HRNET_48.STAGE4.BLOCK = 'BASIC'
HRNET_48.STAGE4.FUSE_METHOD = 'SUM'


# configs for HRNet32
HRNET_32 = CN()
HRNET_32.PRETRAINED_LAYERS = ['*']
HRNET_32.STEM_INPLANES = 64
HRNET_32.FINAL_CONV_KERNEL = 1
HRNET_32.WITH_HEAD = True

HRNET_32.STAGE2 = CN()
HRNET_32.STAGE2.NUM_MODULES = 1
HRNET_32.STAGE2.NUM_BRANCHES = 2
HRNET_32.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_32.STAGE2.NUM_CHANNELS = [32, 64]
HRNET_32.STAGE2.BLOCK = 'BASIC'
HRNET_32.STAGE2.FUSE_METHOD = 'SUM'

HRNET_32.STAGE3 = CN()
HRNET_32.STAGE3.NUM_MODULES = 4
HRNET_32.STAGE3.NUM_BRANCHES = 3
HRNET_32.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_32.STAGE3.NUM_CHANNELS = [32, 64, 128]
HRNET_32.STAGE3.BLOCK = 'BASIC'
HRNET_32.STAGE3.FUSE_METHOD = 'SUM'

HRNET_32.STAGE4 = CN()
HRNET_32.STAGE4.NUM_MODULES = 3
HRNET_32.STAGE4.NUM_BRANCHES = 4
HRNET_32.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_32.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HRNET_32.STAGE4.BLOCK = 'BASIC'
HRNET_32.STAGE4.FUSE_METHOD = 'SUM'


# configs for HRNet18
HRNET_18 = CN()
HRNET_18.PRETRAINED_LAYERS = ['*']
HRNET_18.STEM_INPLANES = 64
HRNET_18.FINAL_CONV_KERNEL = 1
HRNET_18.WITH_HEAD = True

HRNET_18.STAGE2 = CN()
HRNET_18.STAGE2.NUM_MODULES = 1
HRNET_18.STAGE2.NUM_BRANCHES = 2
HRNET_18.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
HRNET_18.STAGE2.BLOCK = 'BASIC'
HRNET_18.STAGE2.FUSE_METHOD = 'SUM'

HRNET_18.STAGE3 = CN()
HRNET_18.STAGE3.NUM_MODULES = 4
HRNET_18.STAGE3.NUM_BRANCHES = 3
HRNET_18.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
HRNET_18.STAGE3.BLOCK = 'BASIC'
HRNET_18.STAGE3.FUSE_METHOD = 'SUM'

HRNET_18.STAGE4 = CN()
HRNET_18.STAGE4.NUM_MODULES = 3
HRNET_18.STAGE4.NUM_BRANCHES = 4
HRNET_18.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
HRNET_18.STAGE4.BLOCK = 'BASIC'
HRNET_18.STAGE4.FUSE_METHOD = 'SUM'

# configs for HRNet2x20
HRNET2X_20 = CN()
HRNET2X_20.FINAL_CONV_KERNEL = 1

HRNET2X_20.STAGE1 = CN()
HRNET2X_20.STAGE1.NUM_MODULES = 1
HRNET2X_20.STAGE1.NUM_BRANCHES = 2
HRNET2X_20.STAGE1.NUM_BLOCKS = [4, 4]
HRNET2X_20.STAGE1.NUM_CHANNELS = [32, 64]
HRNET2X_20.STAGE1.BLOCK = 'BOTTLENECK'
HRNET2X_20.STAGE1.FUSE_METHOD = 'SUM'

HRNET2X_20.STAGE2 = CN()
HRNET2X_20.STAGE2.NUM_MODULES = 1
HRNET2X_20.STAGE2.NUM_BRANCHES = 3
HRNET2X_20.STAGE2.NUM_BLOCKS = [4, 4, 4]
HRNET2X_20.STAGE2.NUM_CHANNELS = [20, 40, 80]
HRNET2X_20.STAGE2.BLOCK = 'BASIC'
HRNET2X_20.STAGE2.FUSE_METHOD = 'SUM'

HRNET2X_20.STAGE3 = CN()
HRNET2X_20.STAGE3.NUM_MODULES = 4
HRNET2X_20.STAGE3.NUM_BRANCHES = 4
HRNET2X_20.STAGE3.NUM_BLOCKS = [4, 4, 4, 4]
HRNET2X_20.STAGE3.NUM_CHANNELS = [20, 40, 80, 160]
HRNET2X_20.STAGE3.BLOCK = 'BASIC'
HRNET2X_20.STAGE3.FUSE_METHOD = 'SUM'

HRNET2X_20.STAGE4 = CN()
HRNET2X_20.STAGE4.NUM_MODULES = 3
HRNET2X_20.STAGE4.NUM_BRANCHES = 5
HRNET2X_20.STAGE4.NUM_BLOCKS = [4, 4, 4, 4, 4]
HRNET2X_20.STAGE4.NUM_CHANNELS = [20, 40, 80, 160, 320]
HRNET2X_20.STAGE4.BLOCK = 'BASIC'
HRNET2X_20.STAGE4.FUSE_METHOD = 'SUM'

MODEL_CONFIGS = {
    'hrnet18': HRNET_18,
    'hrnet32': HRNET_32,
    'hrnet48': HRNET_48,
    'hrnet64': HRNET_64,
    'hrnet2x20': HRNET2X_20,
}
