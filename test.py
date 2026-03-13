import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.nets import SwinUNETR
def count_parameters():
    # 1. 定义模型参数 (保持与你创建模型时一致)
    spatial_dims = 3          # 如果是2D数据请改为2
    in_channels = 1           # 输入通道数
    out_channels = 1          # 输出通道数
    
    num_res_blocks = (2, 2, 2, 2)
    channels = (32, 64, 64, 64)
    attention_levels = (False, False, True, True)
    norm_num_groups = 8
    with_conditioning = False
    resblock_updown = True
    num_head_channels = 8
    use_flash_attention = False 
    # 注意：这里建议设为 False 来计算参数。
    # 原因：Flash Attention 只是加速算子，不改变参数量。
    # 设为 False 可以避免因未安装 flash-attn 库而导致的初始化报错。

    print("正在初始化模型...")
    
    try:
        # 2. 实例化模型
        # model = DiffusionModelUNet(
        #         spatial_dims=3,
        #         in_channels=2,
        #         out_channels=1,
        #         num_res_blocks=(2, 2, 2, 2),
        #         channels=(32, 64, 64, 64),
        #         attention_levels=(False, False, True, True),
        #         norm_num_groups=8,
        #         with_conditioning=False,
        #         resblock_updown=True,
        #         num_head_channels=8,
        #         use_flash_attention=True,
        #     )
        
        model = SwinUNETR(
        in_channels=2,
        out_channels=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        downsample="mergingv2",
        use_v2=True,
    )
        print("模型初始化成功！\n")

    except Exception as e:
        print(f"模型初始化失败: {e}")
        print("提示：如果报错关于 'flash_attn'，请确保已安装 flash-attn 或将 use_flash_attention 设为 False。")
        return

    # 3. 计算参数量
    # total_params: 所有参数的数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # trainable_params: 需要梯度的参数数量 (通常 diffusion unet 全部可训练)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 辅助函数：格式化数字 (例如 1,024,000 -> 1.02 M)
    def format_number(num):
        if num >= 1e6:
            return f"{num / 1e6:.2f} M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f} K"
        else:
            return str(num)

    # 4. 打印结果
    print("-" * 50)
    print(f"模型架构: DiffusionModelUNet ({spatial_dims}D)")
    # 直接打印我们定义的变量，而不是从 model 对象获取
    print(f"配置详情:")
    print(f"  - Channels: {channels}")
    print(f"  - Num Res Blocks: {num_res_blocks}")
    print(f"  - Attention Levels: {attention_levels}")
    print(f"  - In/Out Channels: {in_channels} -> {out_channels}")
    print("-" * 50)
    print(f"总参数量 (Total Parameters):     {total_params:,}  ({format_number(total_params)})")
    print(f"可训练参数量 (Trainable Params): {trainable_params:,}  ({format_number(trainable_params)})")
    
    # 估算显存占用 (浮点数为 4 bytes)
    memory_mb = (total_params * 4) / (1024 ** 2)
    print(f"理论权重显存占用 (FP32):         ~{memory_mb:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    count_parameters()