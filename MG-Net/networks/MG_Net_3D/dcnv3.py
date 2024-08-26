from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes##(2,8,8,96)
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1#6
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1#6

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))#(6,6)(6,6)
    ref_y = ref_y.reshape(-1)[None] / H_#(1,36)
    ref_x = ref_x.reshape(-1)[None] / W_#(1,36)

    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)#(1,6,6,1,2)

    return ref
def _get_reference_points_3d(spatial_shapes, device, kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w, pad_d=0, pad_h=0, pad_w=0, stride_d=1, stride_h=1, stride_w=1):
    _, D_, H_, W_, _ = spatial_shapes## (2,8,8,8,96)
    D_out = (D_ - (dilation_d * (kernel_d - 1) + 1)) // stride_d + 1
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_z, ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_d * (kernel_d - 1)) // 2 + 0.5,
            (dilation_d * (kernel_d - 1)) // 2 + 0.5 + (D_out - 1) * stride_d,
            D_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device)
    )#(6,6,6)

    ref_z = ref_z.reshape(-1)[None] / D_#(1,216)
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y, ref_z), -1).reshape(
        1, D_out, H_out, W_out, 1, 3)#(1,6,6,6,1,3)

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes#(2,8,8,96)
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device))#(3,3)(3,3)

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)#(4,9,2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)#(1,1,1,36,2)

    return grid
def _generate_dilation_grids_3d(spatial_shapes, kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w, group, device):
    _, D_, H_, W_, _ = spatial_shapes#(2,8,8,8,96)
    points_list = []
    
    x, y, z = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_d * (kernel_d - 1)) // 2),
            -((dilation_d * (kernel_d - 1)) // 2) + (kernel_d - 1) * dilation_d, kernel_d,
            dtype=torch.float32,
            device=device)
    )#(3,3,3)
    
    points_list.extend([x / W_, y / H_, z / D_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 3).repeat(1, group, 1).permute(1, 0, 2)#(4,27,3)
    grid = grid.reshape(1, 1, 1, group * kernel_d * kernel_h * kernel_w, 3)#(1,1,1,108,3)

    return grid


def dcnv3_core_pytorch(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    # for debug and test only,
    # need to use cuda version instead
    input = F.pad(
        input,#(2,6,6,96)
        [0, 0, pad_h, pad_h, pad_w, pad_w])#(2,8,8,96)
    N_, H_in, W_in, _ = input.shape#(2,8,8,96)
    _, H_out, W_out, _ = offset.shape#(2,8,8,72)

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)#(1,6,6,1,2)
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)#(1,1,1,36,2)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*kernel_h*kernel_w).to(input.device)#(1,1,1,72)

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
        offset * offset_scale / spatial_norm#(2,6,6,72)

    P_ = kernel_h * kernel_w#9
    sampling_grids = 2 * sampling_locations - 1#(2,6,6,72)
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in)#(8,24,8,8)
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)#(8,36,+-,2)
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)#(8,24,36,9)

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)#(8,1,36,9)
    output = (sampling_input_ * mask).sum(-1).view(N_,
                                                   group*group_channels, H_out*W_out)#(2,96,36)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()
def dcnv3_core_pytorch_3d(
        input, offset, mask, kernel_d, kernel_h,
        kernel_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w, dilation_d, dilation_h,
        dilation_w, group, group_channels, offset_scale):
    # 针对三维图像进行填充
    input = F.pad(
        input,# (2,6,6,6,96)
        [0, 0, pad_w, pad_w, pad_h, pad_h, pad_d, pad_d]
    )# (2,8,8,8,96)
    N_, D_in, H_in, W_in, _ = input.shape  # (2,8,8,8,96)
    _, D_out, H_out, W_out, _ = offset.shape  # (2,6,6,6,216)

    ref = _get_reference_points_3d(input.shape, input.device, kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)#(1,6,6,6,1,3)
    grid = _generate_dilation_grids_3d(input.shape, kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w, group, input.device)#(1,1,1,108,3)
    spatial_norm = torch.tensor([W_in, H_in, D_in]).reshape(1, 1, 1, 3).\
        repeat(1, 1, 1, group * kernel_d * kernel_h * kernel_w).to(input.device)#(1,1,1,324)

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1,1).flatten(4, 5) + \
        offset * offset_scale / spatial_norm#(2,6,6,6,324)

    P_ = kernel_d * kernel_h * kernel_w#27
    sampling_grids = 2 * sampling_locations - 1#(2,6,6,6,324)
    input_ = input.view(N_, D_in * H_in * W_in, group * group_channels).transpose(1, 2).\
        reshape(N_ * group, group_channels, D_in, H_in, W_in)#(8,24,8,8,8)
    sampling_grid_ = sampling_grids.view(N_, D_out * H_out * W_out, group, P_, 1,3).transpose(1, 2).\
        flatten(0, 1)#(8,216,27,3)
    
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False
    )#(8,24,216,27,1)
    sampling_input_=sampling_input_.reshape(N_*group, group_channels,D_out * H_out * W_out,P_)#(8,24,216,27)

    mask = mask.view(N_, D_out * H_out * W_out, group, P_).transpose(1, 2).\
        reshape(N_ * group, 1, D_out * H_out * W_out, P_)#(8,1,216,27)
    output = (sampling_input_ * mask).sum(-1).view(N_,group*group_channels, D_out * H_out*W_out)#(2,96,216)

    return output.transpose(1, 2).reshape(N_, D_out, H_out, W_out, -1).contiguous()#(2,6,6,6,96)

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)
class to_channels_first_3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 4, 1, 2,3)

class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)
class to_channels_last_3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 4, 1)

def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)
def build_norm_layer_3d(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first_3d())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last_3d())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last_3d())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first_3d())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,#(2,6,6,96)
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale#(2,6,6,4)


class DCNv3_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape#(2,6,6,96)

        x = self.input_proj(input)#(2,6,6,96)
        x_proj = x#(2,6,6,96)

        x1 = input.permute(0, 3, 1, 2)#(2,96,6,6)
        x1 = self.dw_conv(x1)#(2,6,6,96)
        offset = self.offset(x1)#(2,6,6,72)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)#(2,6,6,4,9)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)#(2,6,6,36)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,# 3 3
            self.stride, self.stride,#1 1
            self.pad, self.pad,#1 1
            self.dilation, self.dilation,#1 1
            self.group, self.group_channels,#4 24
            self.offset_scale)#1  #x(2,6,6,96)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)#(2,6,6,96)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale#(2,6,6,96)
        x = self.output_proj(x)#(2,6,6,96)

        return x
class DCNv3_pytorch_3d(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer_3d(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * kernel_size* 3)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size* kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, D, H, W, C)  # 修改这里，添加深度维度
        :return output                     (N, D, H, W, C)  # 修改这里，添加深度维度
        """
        N, D, H, W, _ = input.shape  # (2,6,6,6,96)

        x = self.input_proj(input)# (2,6,6,6,96)
        x_proj = x# (2,6,6,6,96)

        x1 = input.permute(0, 4, 1, 2, 3) #(2,96,6,6,6)
        x1 = self.dw_conv(x1)# (2,6,6,6,96)
        offset = self.offset(x1)# (2,6,6,6,216)
        mask = self.mask(x1).reshape(N, D, H, W, self.group, -1)# (2,6,6,6,4,27)
        mask = F.softmax(mask, -1).reshape(N, D, H, W, -1)# (2,6,6,6,108)

        x = dcnv3_core_pytorch_3d(
            x, offset, mask,
            self.kernel_size, self.kernel_size,self.kernel_size,# 3 3 3
            self.stride, self.stride,self.stride,#1 1 1
            self.pad, self.pad,self.pad,#1 1 1
            self.dilation, self.dilation,self.dilation,#1 1 1
            self.group, self.group_channels,#4 24
            self.offset_scale)#1  #
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, 1,self.channels // self.group).flatten(-2)#(2,6,6,6,96)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale#(2,6,6,6,96)
        x = self.output_proj(x)#(2,6,6,6,96)

        return x

if __name__ == "__main__":
    # 创建一个 DCNv3 模型的实例
    dcnv3_model = DCNv3_pytorch_3d(
        channels=96,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        center_feature_scale=True
    )

    # 创建一个示例输入（这里仅仅是一个形状的示例，实际输入应根据模型要求进行调整）
    input_tensor = torch.randn(2, 24, 24, 24,96)

    # 调用模型进行前向传播
    output = dcnv3_model(input_tensor)

    # 打印输出结果
    print("Model output shape:", output.shape)