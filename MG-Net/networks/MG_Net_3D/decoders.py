import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

import math
from PIL import Image
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc 

from networks.MG_Net_3D.gcn_lib import Grapher3D as GCB3D
from networks.MG_Net_3D.gcn_lib import Grapher as GCB

class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    
    
if __name__ == "__main__":
    # # 创建一个简单的输入张量（例如，大小为 [batch_size, channels, height, width]）
    # input_tensor = torch.randn(2, 768, 6,6)
    input_tensor = torch.randn(2, 768, 6,6,6)
    print("Input Tensor:")
    print(input_tensor.shape)
    # 创建 SPA 模型
    # spa_model = SPA(kernel_size=7)
    # # 进行前向传播
    # output = spa_model(input_tensor)
    # output=output*input_tensor
    # 打印输出结果
    # GCB_model=GCB(768, 11, 1, 'mr', 'gelu', 'batch',True, False, 0.2, 1, n=6*6, drop_path=0.0,relative_pos=True, padding=5)
    GCB_model=GCB3D(768, 11, 1, 'mr', 'gelu', 'batch',True, False, 0.2, 1, n=6*6*6, drop_path=0.0,relative_pos=True, padding=5)
    output=GCB_model(input_tensor)
    
    print("\nOutput Tensor:")
    # print(output)
    print(output.shape)
