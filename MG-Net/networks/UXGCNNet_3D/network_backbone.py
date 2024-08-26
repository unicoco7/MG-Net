#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
# from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from networks.UXGCNNet_3D.CGCN_UpBlock import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.UXGCNNet_3D.uxnet_encoder import uxnet_conv
from networks.UXGCNNet_3D.swin_transformer import *
from networks.UXGCNNet_3D.decoders import *
from networks.UXGCNNet_3D.skipconnect3d import *

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class UXGCNNET(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        # self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        self.normalize = True
        img_size=(96, 96, 96)
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = SwinTransformer(
            in_chans=self.in_chans,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
        )

        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)
        self.spa = SPA()
        self.gcb3D_5=GCB3D(feat_size[3]*2, 11, 1, 'mr', 'gelu', 'batch',True, False, 0.2, 1, n=6*6*6, drop_path=0.0,relative_pos=True, padding=5)
        self.gcb3D_4=GCB3D(feat_size[2]*2, 11, 1, 'mr', 'gelu', 'batch',True, False, 0.2, 1, n=6*6*6, drop_path=0.0,relative_pos=True, padding=5)
        self.gcb3D_3=GCB3D(feat_size[1]*2, 11, 1, 'mr', 'gelu', 'batch',True, False, 0.2, 1, n=6*6*6, drop_path=0.0,relative_pos=True, padding=5)
        # self.gcb3D_2=GCB3D(feat_size[0]*2, 11, 1, 'mr', 'gelu', 'batch',True, False, 0.2, 1, n=6*6*6, drop_path=0.0,relative_pos=True, padding=5)
        self.out_head3 = nn.Conv3d(feat_size[3], self.out_chans, 1)
        self.out_head2 = nn.Conv3d(feat_size[2], self.out_chans, 1)
        self.out_head1 = nn.Conv3d(feat_size[1], self.out_chans, 1)

        
        backbone_feature_shape={'res3': {'channel': 96, 'stride': 8}, 'res4': {'channel': 192, 'stride': 16}, 'res5': {'channel': 384, 'stride': 32}}
        self.detr_decoder = MSDeformAttnPixelDecoder3D(input_shape=backbone_feature_shape)
        self.detr_output_enc2=nn.Conv3d(120, self.feat_size[1], 1)
        self.detr_output_enc3=nn.Conv3d(120, self.feat_size[2], 1)
        self.detr_output_enc4=nn.Conv3d(120, self.feat_size[3], 1)


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        # hidden_states_out = self.swinViT(x_in, self.normalize)
        enc1 = self.encoder1(x_in)
        # print(enc1.size())
        x2 = outs[0]#+hidden_states_out[0]
        enc2 = self.encoder2(x2)
        # print(enc2.size())
        x3 = outs[1]#+hidden_states_out[1]
        enc3 = self.encoder3(x3)
        # print(enc3.size())
        x4 = outs[2]#+hidden_states_out[2]
        enc4 = self.encoder4(x4)
        # print(enc4.size())
        # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
        x5=outs[3]#+hidden_states_out[3]
        
        # #detr_transformer_block 2
        input_tensor = {"res3": enc2,"res4": enc3,"res5": enc4}
        detr_output=self.detr_decoder(input_tensor)
        detr_output_enc2=self.detr_output_enc2(detr_output[0])
        detr_output_enc3=self.detr_output_enc3(detr_output[1])
        detr_output_enc4=self.detr_output_enc4(detr_output[2])
        enc4=enc4+detr_output_enc4
        enc3=enc3+detr_output_enc3
        enc2=enc2+detr_output_enc2



        enc_hidden = self.encoder5(x5)
        enc_hidden=self.gcb3D_5(enc_hidden) #gcb_
        enc_hidden=self.spa(enc_hidden)*enc_hidden  #spa_
        dec3 = self.decoder5(enc_hidden, enc4)
        dec3=self.gcb3D_4(dec3) #gcb_
        dec3=self.spa(dec3)*dec3#spa_
        dec2 = self.decoder4(dec3, enc3)
        dec2=self.gcb3D_3(dec2) #gcb_
        dec2=self.spa(dec2)*dec2#spa_
        dec1 = self.decoder3(dec2, enc2)
        # dec1=self.gcb3D_2(dec1) #gcb
        # dec1=self.spa(dec1)*dec1#spa
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        out=self.out(out)
        
        #muti_stage_output 3
        p13=self.out_head3(dec3)
        p12=self.out_head2(dec2)
        p11=self.out_head1(dec1)

        p13 = F.interpolate(p13, size=(96, 96, 96),  mode='trilinear', align_corners=False)
        p12 = F.interpolate(p12, size=(96, 96, 96),  mode='trilinear', align_corners=False)
        p11 = F.interpolate(p11, size=(96, 96, 96),  mode='trilinear', align_corners=False)

        out=(out+p13+p12+p11)/4
        
        return out