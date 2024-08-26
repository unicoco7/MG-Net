# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# modified from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# relative position embedding
# References: https://arxiv.org/abs/2009.13658
# --------------------------------------------------------
def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    # print(embed_dim,grid_size)#768 6
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    # print("pos_embed.shape:",pos_embed.shape)#(36, 768)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    # print("relative_pos.shape:",relative_pos.shape)#(36, 36)
    return relative_pos
def get_3d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: tuple of grid depth, height, and width
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim]
    """
    # print(embed_dim, grid_size)#768 6
    pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size)
    # print("pos_embed.shape:",pos_embed.shape)#(216, 768)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    # print("relative_pos.shape:",relative_pos.shape)#(216, 216)
    return relative_pos

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)#(6,)
    grid_w = np.arange(grid_size, dtype=np.float32)#(6,)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first(2,6,6)
    grid = np.stack(grid, axis=0)#(2,6,6)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)#(36, 768)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple of grid depth, height, and width
    return:
    pos_embed: [grid_size[0]*grid_size[1]*grid_size[2], embed_dim] or [1+grid_size[0]*grid_size[1]*grid_size[2], embed_dim] (w/ or w/o cls_token)
    """
    grid_d = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)#768
    # print("grid.shape:",grid.shape)# (3, 1, 6, 6, 6)
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    # print("pos_embed.shape:",pos_embed.shape)#(2744, 1152)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # print(embed_dim,grid.shape)#768 (2, 1, 6, 6)
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)(36, 384)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)(36, 384)
    
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)(36, 768)

    return emb
def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, D/3)(216, 256)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (D*H*W, D/3)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1) # (D*H*W, D)#(216, 768)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)(36, 384)

    return emb

