from natten import NeighborhoodAttention3D
import torch

if __name__ == "__main__":

    # 使用 NeighborhoodAttention3D
    na3d = NeighborhoodAttention3D(dim=384, kernel_size=3, dilation=2, num_heads=8)
    input_3d = torch.randn(2, 6, 6, 6, 384)  # 输入大小为 [batch_size, depth, height, width, dim]
    output_3d = na3d(input_3d)
    print(output_3d.shape)
    # # NA3D 还支持不同的 kernel size 和 dilation 值
    # na3d_custom = NeighborhoodAttention3D(
    #     dim=128,
    #     kernel_size=7,
    #     kernel_size_d=5,
    #     dilation=2,
    #     dilation_d=3,
    #     num_heads=4
    # )
    # input_3d_custom = torch.randn(2, 384, 6, 6, 6, 128)
    # output_3d_custom = na3d_custom(input_3d_custom)
