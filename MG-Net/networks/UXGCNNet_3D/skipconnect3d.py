import numpy as np
import math
import copy
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    @value: bs, sum(h, w), num_head, dim
    @sampling_locations: bs, sum(h, w), num_head, num_layer, 4, 2
    @attention_weights: bs, sum(h, w), num_head, num_layer, 4
    """
    N_, S_, M_, Dim = value.shape#(2,15768,8,30)
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape#(2,15768,8,3,4,3)
     # Split the input 'value' tensor based on the spatial shapes
    # Split the input 'value' tensor based on the spatial shapes
    value_list = value.split([D_ * H_ * W_ for D_, H_, W_ in value_spatial_shapes], dim=1)#(2,216,8,30)(2,1728,8,30)(2,13824,8,30)

    # Transform sampling locations to the range [-1, 1]
    sampling_grids = 2 * sampling_locations - 1#(2,15768,8,3,4,3)

    # Initialize an empty list to store the sampled values
    sampling_value_list = []

    for lid_, (D_, H_, W_) in enumerate(value_spatial_shapes):
        # Flatten and transpose the 'value' tensor for grid sampling
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, Dim, D_, H_, W_)#(16,30,6,6,6)
        # Extract and flatten the sampling grid for the current layer
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]#(2,15768,8,4,3)
        sampling_grid_l_ = sampling_grid_l_.transpose(1, 2).flatten(0, 1)#(16,15768,4,3)
        # Perform grid sampling using bilinear interpolation for 3D data
        sampling_grid_l_ = sampling_grid_l_.unsqueeze(-2)#(16,15768,4,1,3)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)#(16,30,15768,4,1)
        sampling_value_l_ = sampling_value_l_.squeeze(-1)#(16,30,15768,4)
        # Append the sampled values to the list
        sampling_value_list.append(sampling_value_l_)

    # Reshape and transpose attention weights for element-wise multiplication
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)#(16,1,15768,12)
    # Element-wise multiplication and sum to get the final output
    output = (torch.stack(sampling_value_list, dim=-2).squeeze(2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*Dim, Lq_)#(2,240,15768)

    # Transpose the output for contiguous memory layout
    return output.transpose(1, 2).contiguous()

class MSDeformAttn3D(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module for 3D data
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)  # Adjusted for 3D
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # for i in range(self.n_points):
        #     grid_init[:, :, i, :] *= i + 1
        # with torch.no_grad():
        #     self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape#(2,15768,240)
        N, Len_in, _ = input_flatten.shape#(2,15768,240)
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)#(2,15768,240)
        # if input_padding_mask is not None:#(2,15768)
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)#(2,15768,240)

        # Adjusted for 3D
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 3)#(2,15768,8,3,4,3)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)#(2,15768,8,12)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)#(2,15768,8,3,4)

        offset_normalizer = torch.stack([input_spatial_shapes[..., 2], input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)#(3,3)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]#(2,15768,8,3,4,3)

        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


# MSDeformAttn Transformer encoder in deformable detr
# MSDeformAttn Transformer encoder for 3D data
class MSDeformAttnTransformerEncoderLayer3D(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn3D(d_model, n_levels, n_heads, n_points)  # Assuming you have a 3D version of your custom attention
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder3D(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):
            ref_z, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                                                 torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                                 torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))#(6,6,6)(6,6,6)(6,6,6)
            ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D_)#(2,216)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)#(2,216)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)#(2,216)
            ref = torch.stack((ref_z, ref_y, ref_x), -1)  # [1, D_ * H_ * W_, 3]
            reference_points_list.append(ref)#(2,216,3)(2,1728,3)(2,13824,3)
        reference_points = torch.cat(reference_points_list, 1)#(2,15768,3)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]#(2,15768,3,3)
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src#(2,15768,240)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)#(2,15768,3,3)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)#(2,756,256)

        return output

class MSDeformAttnTransformerEncoderOnly3D(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer3D(d_model, dim_feedforward,
                                                             dropout, activation,
                                                             num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder3D(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn3D):  # Assuming you have a 3D version of your custom attention
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape  # Assuming mask has a depth dimension#(2,6,6,6)
        valid_D = torch.sum(~mask[:, 0, :, :], (1, 2))#(2)
        valid_H = torch.sum(~mask[:, :, 0, :], (1, 2))#(2)
        valid_W = torch.sum(~mask[:, :, :, 0], (1, 2))#(2)
        valid_ratio_d = valid_D.float() / D#(2)
        valid_ratio_h = valid_H.float() / H#(2)
        valid_ratio_w = valid_W.float() / W#(2)
        valid_ratio = torch.stack([valid_ratio_d, valid_ratio_h, valid_ratio_w], -1)#(2,3)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3), x.size(4)), device=x.device, dtype=torch.bool) for x in srcs]
        #(2,6,6,6)(2,12,12,12)(2,24,24,24)
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, d, h, w = src.shape#(2,240,6,6,6)
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)#(6,6,6)(12,12,12)(24,24,24)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)#(2,216)(2,1728)(2,13826)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)#(2,216,240)(2,1728,240)(2,13824,240)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)#(2,216,240)(2,1728,240)(2,13824,240)
            src_flatten.append(src)#(2,216,240)(2,1728,240)(2,13824,240)
            mask_flatten.append(mask)#(2,216)(2,1728)(2,13824)
        src_flatten = torch.cat(src_flatten, 1)#(2,15768,240)
        mask_flatten = torch.cat(mask_flatten, 1)#(2,15768)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)#(2,15768,240)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)#(3,3)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))#(3)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)#(2,3,3)
        # spatial_shapes=torch.tensor([[6,6],[12,12],[24,24]])
        # valid_ratios=torch.ones((2, 3, 2))
        # print(valid_ratios.shape)
        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index
        return

class PositionEmbeddingSine3D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)#x(2,768,6,6,6)
        not_mask = ~mask#(2,6,6,6)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)#(2,6,6,6)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)#(2,6,6,6)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)#(2,6,6,6)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :, :] + eps) * self.scale#(2,6,6,6)
            x_embed = x_embed / (x_embed[:, :, -1:, :] + eps) * self.scale#(2,6,6,6)
            z_embed = z_embed / (z_embed[:, :, :, -1:] + eps) * self.scale#(2,6,6,6)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_y, pos_x, pos_z), dim=4).permute(0, 4, 1, 2, 3)

        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class MSDeformAttnPixelDecoder3D(nn.Module):
    def __init__(
        self,
        input_shape,
        transformer_dropout=0.1,
        transformer_nheads=8,
        transformer_dim_feedforward=2048,
        transformer_enc_layers=3,
        conv_dim=120,
        mask_dim=256,
        transformer_in_features=["res3", "res4", "res5"],
        common_stride=4,
    ):
        super(MSDeformAttnPixelDecoder3D, self).__init__()

        # Define your layers and modules here
        self.conv_dim = conv_dim

        # Example input tensor shape: {"res2": (2, 96, 48, 48), "res3": (2, 192, 24, 24), ...}
        self.input_shape = input_shape
        self.channel=[input_shape["res3"]['channel'],input_shape["res4"]['channel'],input_shape["res5"]['channel']]
        # Assuming input_shape contains "res3", "res4", "res5"
        self.decoder_res3 = self.build_decoder_block(self.channel[0], conv_dim, kernel_size=1)
        self.decoder_res4 = self.build_decoder_block(self.channel[1], conv_dim, kernel_size=1)
        self.decoder_res5 = self.build_decoder_block(self.channel[2], conv_dim, kernel_size=1)

        N_steps = conv_dim // 3
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)

        self.transformer_num_feature_levels=len(transformer_in_features)#3
        self.transformer = MSDeformAttnTransformerEncoderOnly3D(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )

    def build_decoder_block(self, in_channels, out_channels, kernel_size=1):
        block =nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size),
            nn.GroupNorm(6, out_channels),
            nn.ReLU(inplace=True),
        )
        for layer in block:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)

        return block

    def forward(self, input_tensor):
        # Extract features from the input tensor
        res3_features = input_tensor["res3"]#(2, 192, 24, 24,24)
        res4_features = input_tensor["res4"]#(2, 384, 12, 12,12)
        res5_features = input_tensor["res5"]#(2, 768, 6, 6,6)

        # Apply the decoder blocks to res3, res4, and res5 features
        srcs_res5 = self.decoder_res5(res5_features)
        srcs_res4 = self.decoder_res4(res4_features)
        srcs_res3 = self.decoder_res3(res3_features)        
        srcs = [srcs_res5,srcs_res4,srcs_res3]#(2,240,6,6,6) (2,240,12,12,12)(2,240,24,24,24)

        pos_res5=self.pe_layer(res5_features)#torch.Size([2, 240, 6, 6, 6])
        pos_res4=self.pe_layer(res4_features)#torch.Size([2, 240, 12, 12, 12])
        pos_res3=self.pe_layer(res3_features)#torch.Size([2, 240, 24, 24, 24])
        pos = [pos_res5,pos_res4,pos_res3]

        y, spatial_shapes, level_start_index =self.transformer(srcs, pos)#([2, 15768, 240])([3, 3])([3])
        bs = y.shape[0]#2

        split_size_or_sections = [None] * self.transformer_num_feature_levels#(none,none,none)
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)#(2,216,240)(2,1728,240)(2,13824,240)


        out = []
        channel=self.channel[::-1]
        for i, z in enumerate(y):
            # z=nn.Linear(self.conv_dim, channel[i])(z)
            out_i=z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1], spatial_shapes[i][2])#(2,240,6,6,6)(2,240,12,12,12)(2,240,24,24,24)
            # out_i= self.build_decoder_block(self.conv_dim, channel[i], kernel_size=1)(out_i)
            out.append(out_i)

        return out[::-1]

if __name__ == "__main__":
    # Example usage
    input_tensor = {
        "res3": torch.randn(2, 96, 48, 48,48),
        "res4": torch.randn(2, 192, 24, 24,24),
        "res5": torch.randn(2, 384, 12, 12,12),
        # "res5": torch.randn(2, 768, 6, 6,6),
    }
    # backbone_feature_shape = dict()
    # channels = [96, 192, 384,768]
    # for i, channel in enumerate(channels):
    #     backbone_feature_shape[f'res{i+2}'] = Dict({'channel': channel, 'stride': 2**(i+2)})
    # backbone_feature_shape={'res2': {'channel': 96, 'stride': 4}, 'res3': {'channel': 192, 'stride': 8}, 'res4': {'channel': 384, 'stride': 16}, 'res5': {'channel': 768, 'stride': 32}}

    backbone_feature_shape={'res3': {'channel': 96, 'stride': 8}, 'res4': {'channel': 192, 'stride': 16}, 'res5': {'channel': 384, 'stride': 32}}

    # Instantiate the MSDeformAttnPixelDecoder
    decoder = MSDeformAttnPixelDecoder3D(input_shape=backbone_feature_shape)

    # Forward pass
    out= decoder(input_tensor)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

