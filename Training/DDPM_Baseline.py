import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from monai.networks.nets import DiffusionModelUNet


class DDPMScheduler:
    """DDPM噪声调度器"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 用于采样的计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(self, original, noise, timesteps):
        """添加噪声到原始图像"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        
        noisy = sqrt_alpha_prod.to(original.device) * original + \
                sqrt_one_minus_alpha_prod.to(original.device) * noise
        return noisy
    
    def step(self, model_output, timestep, sample):
        """单步去噪"""
        t = timestep
        
        # 预测原始样本
        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t].to(sample.device) * model_output
        ) / self.sqrt_alphas_cumprod[t].to(sample.device)
        
        # 计算前一步的样本
        pred_sample_direction = self.sqrt_one_minus_alphas_cumprod[t].to(sample.device) * model_output
        
        prev_sample = (
            self.sqrt_recip_alphas[t].to(sample.device) * 
            (sample - self.betas[t].to(sample.device) / self.sqrt_one_minus_alphas_cumprod[t].to(sample.device) * model_output)
        )
        
        # 添加噪声（除了最后一步）
        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(self.posterior_variance[t].to(sample.device))
            prev_sample = prev_sample + variance * noise
        
        return prev_sample


class CTtoPETDiffusion:

    def __init__(
        self,
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        num_res_blocks=1,           # 减少残差块
        channels=(8, 8),          # 只有2层！
        attention_levels=(False, False),  # 长度=2
        num_train_timesteps=1000,
        device='cuda'
    ):
        self.device = device
        self.model = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            channels=channels,
            attention_levels=attention_levels,
            norm_num_groups=8,       # 必须整除最小 channel (16)
            with_conditioning=False,
            resblock_updown=True,
            num_head_channels=8,     # 虽然不用 attention，但需设小值
            
            use_flash_attention = True
        ).to(device)
        print(count_parameters(model=self.model))
        # 初始化调度器
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.device=device
        
    def train_step(self, ct_images, pet_images, optimizer):
        """单步训练"""
        self.model.train()
        batch_size = ct_images.shape[0]
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,)
        ).long()
        
        # 生成随机噪声
        noise = torch.randn_like(pet_images)
        
        # 添加噪声到PET图像
        noisy_pet = self.scheduler.add_noise(pet_images, noise, timesteps)
        
        # 将CT和noisy PET拼接作为输入
        model_input = torch.cat([ct_images, noisy_pet], dim=1)
        
        # 预测噪声
        noise_pred = self.model(model_input.to(self.device), timesteps.to(self.device))
        
        # 计算损失（MSE）
        loss = nn.functional.mse_loss(noise_pred, noise.to(self.device))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def generate(self, ct_images, num_inference_steps=50):
        """从CT生成PET"""
        self.model.eval()
        batch_size = ct_images.shape[0]
        
        # 从纯噪声开始
        pet_shape = (batch_size, 1, *ct_images.shape[2:])
        sample = torch.randn(pet_shape, device=self.device)
        
        # 设置采样步数
        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0, num_inference_steps
        ).long().to(self.device)
        
        # 迭代去噪
        for t in tqdm(timesteps, desc="Generating PET"):
            # 准备输入
            model_input = torch.cat([ct_images, sample], dim=1)
            timestep_batch = torch.full((batch_size,), t, device=self.device).long()
            
            # 预测噪声
            noise_pred = self.model(model_input, timestep_batch)
            
            # 去噪一步
            sample = self.scheduler.step(noise_pred, t.item(), sample)
        
        return sample
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scheduler_config': {
                'num_train_timesteps': self.scheduler.num_train_timesteps,
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


def train(
    train_loader,
    val_loader=None,
    num_epochs=100,
    learning_rate=1e-4,
    save_dir='./checkpoints',
    device='cuda'
):
    """训练函数"""
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    diffusion = CTtoPETDiffusion(device=device)
    
    # 优化器
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            # 假设batch是(ct, pet)的元组
            ct_images, pet_images = batch
            ct_images = ct_images.to(device)
            pet_images = pet_images.to(device)
            
            loss = diffusion.train_step(ct_images, pet_images, optimizer)
            train_losses.append(loss)
            
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证
        if val_loader is not None:
            val_losses = []
            diffusion.model.eval()
            
            with torch.no_grad():
                for batch in val_loader:
                    ct_images, pet_images = batch
                    ct_images = ct_images.to(device)
                    pet_images = pet_images.to(device)
                    
                    batch_size = ct_images.shape[0]
                    timesteps = torch.randint(
                        0, diffusion.scheduler.num_train_timesteps, (batch_size,),
                        device=device
                    ).long()
                    
                    noise = torch.randn_like(pet_images)
                    noisy_pet = diffusion.scheduler.add_noise(pet_images, noise, timesteps)
                    model_input = torch.cat([ct_images, noisy_pet], dim=1)
                    noise_pred = diffusion.model(model_input, timesteps)
                    
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                diffusion.save(save_dir / 'best_model.pth')
        else:
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}')
        
        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            diffusion.save(save_dir / f'model_epoch_{epoch+1}.pth')
        
        scheduler.step()
    
    # 保存最终模型
    diffusion.save(save_dir / 'final_model.pth')
    
    return diffusion


@torch.no_grad()
def inference(ct_images, model_path, num_inference_steps=50, device='cuda'):
    """推理函数"""
    # 加载模型
    diffusion = CTtoPETDiffusion(device=device)
    diffusion.load(model_path)
    
    # 确保输入在正确的设备上
    if not isinstance(ct_images, torch.Tensor):
        ct_images = torch.from_numpy(ct_images)
    ct_images = ct_images.to(device)
    
    # 如果输入是单个样本，添加batch维度
    if ct_images.ndim == 4:
        ct_images = ct_images.unsqueeze(0)
    
    # 生成PET
    generated_pet = diffusion.generate(ct_images, num_inference_steps=num_inference_steps)
    
    return generated_pet.cpu().numpy()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 使用示例
if __name__ == "__main__":
    """
    # 1. 准备数据加载器（你需要实现这部分）
    # 假设你的数据加载器返回 (ct_images, pet_images)
    # ct_images: (B, 1, D, H, W) - CT图像
    # pet_images: (B, 1, D, H, W) - PET图像
    
    from your_dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=2)
    
    # 2. 训练模型
    diffusion = train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=1e-4,
        save_dir='./checkpoints',
        device='cuda'
    )
    
    # 3. 推理
    # 加载测试CT图像
    test_ct = torch.randn(1, 1, 64, 128, 128)  # 示例数据
    
    # 生成PET
    generated_pet = inference(
        ct_images=test_ct,
        model_path='./checkpoints/best_model.pth',
        num_inference_steps=50,
        device='cuda'
    )
    
    print(f"Generated PET shape: {generated_pet.shape}")
    """
    pass