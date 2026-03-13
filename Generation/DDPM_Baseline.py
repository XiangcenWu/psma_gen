
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from monai.networks.nets import DiffusionModelUNet
import torch.nn.functional as F

import SimpleITK as sitk
from General.save_itk import tensor_to_itk

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
        # sqrt(alpht_bar)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # sqrt(1 - alpht_bar)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(self, original, noise, timesteps):
        """添加噪声到原始图像"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        
        noisy = sqrt_alpha_prod.to(original.device) * original + \
                sqrt_one_minus_alpha_prod.to(original.device) * noise
        return noisy
    
    def step(self, model_output, timestep, sample):
        """单步去噪"""
        t = timestep

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
        num_train_timesteps=1000,
        device='cuda'
    ):
        self.device = device
        self.model = DiffusionModelUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        num_res_blocks=(2, 2, 2, 2),
        channels=(32, 64, 64, 64),
        attention_levels=(False, False, True, True),
        norm_num_groups=8,
        with_conditioning=False,
        resblock_updown=True,
        num_head_channels=8,
        use_flash_attention=True,
    ).to(self.device)

        # 初始化调度器
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.device=device

    
    @torch.no_grad()
    def generate(self, ct_images, num_inference_steps=1000):
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
        for t in timesteps:
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
        if isinstance(checkpoint, dict) and "scheduler_config" in checkpoint:
            scheduler_config = checkpoint["scheduler_config"]
            num_train_timesteps = scheduler_config.get("num_train_timesteps")
            if (
                num_train_timesteps is not None
                and num_train_timesteps != self.scheduler.num_train_timesteps
            ):
                self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")


def map_zero_one_to_minus_one_one(image):
    return image * 2.0 - 1.0

def map_minus_one_one_to_zero_one(image):
    return (image + 1.0) / 2.0
    
    

def get_pair(
    batch,
    input_key,
    target_key,
    device,
    use_fdg_condition=False,
    fdg_key="fdg_pt",
):
    condition = batch[input_key].float().to(device)
    if use_fdg_condition:
        fdg = batch[fdg_key].float().to(device)
        condition = torch.cat([condition, fdg], dim=1)
    target = batch[target_key].float().to(device)
    condition = map_zero_one_to_minus_one_one(condition)
    target = map_zero_one_to_minus_one_one(target)
    return condition, target


def add_noise_3d(diffusion, target, noise, timesteps):
    view_shape = (timesteps.shape[0],) + (1,) * (target.ndim - 1)
    alpha_schedule = diffusion.scheduler.sqrt_alphas_cumprod
    sigma_schedule = diffusion.scheduler.sqrt_one_minus_alphas_cumprod
    schedule_timesteps = timesteps.to(alpha_schedule.device)

    alpha = alpha_schedule[schedule_timesteps].to(target.device).view(view_shape)
    sigma = sigma_schedule[schedule_timesteps].to(target.device).view(view_shape)
    return alpha * target + sigma * noise


def compute_loss(diffusion, condition, target):
    batch_size = target.shape[0]
    timesteps = torch.randint(
        0,
        diffusion.scheduler.num_train_timesteps,
        (batch_size,),
        device=target.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target)
    noisy_target = add_noise_3d(diffusion, target, noise, timesteps)
    model_input = torch.cat([condition, noisy_target], dim=1)
    noise_pred = diffusion.model(model_input, timesteps)
    return F.mse_loss(noise_pred, noise)


def train_epoch(
    diffusion,
    loader,
    optimizer,
    input_key,
    target_key,
    device,
    epoch,
    epochs,
    use_fdg_condition=False,
    fdg_key="fdg_pt",
):
    diffusion.model.train()
    losses = []
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for batch in progress:
        condition, target = get_pair(
            batch,
            input_key,
            target_key,
            device,
            use_fdg_condition,
            fdg_key,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(diffusion, condition, target)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return float(np.mean(losses))


@torch.no_grad()
def run_inference(diffusion, test_loader, args):


    progress = tqdm(enumerate(test_loader), total=len(test_loader), desc="DDPM inference")
    for index, batch in progress:
        batch_size = batch[args.target_key].shape[0]
        if batch_size != 1:
            raise ValueError(f"Expected test batch size 1, got {batch_size}.")


        condition, _ = get_pair(
            batch,
            args.input_key,
            args.target_key,
            args.device,
            args.use_fdg_condition,
            args.fdg_key,
        )
        prediction = diffusion.generate(
            condition,
            num_inference_steps=args.num_inference_steps,
        )
        prediction = map_minus_one_one_to_zero_one(prediction).clamp(0.0, 1.0)

        target = batch[args.target_key].float()


        case_dir = os.path.join(args.output_dir, f"sample_{index:04d}")
        os.makedirs(case_dir, exist_ok=True)

        sitk.WriteImage(tensor_to_itk(prediction), os.path.join(case_dir, 'psma_prediction.nii.gz'))
        sitk.WriteImage(tensor_to_itk(target), os.path.join(case_dir, 'psma_gt.nii.gz'))
