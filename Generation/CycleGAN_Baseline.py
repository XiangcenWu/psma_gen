import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from tqdm import tqdm

from General.save_itk import tensor_to_itk
from Generation.utils import (
    get_pair,
    map_minus_one_one_to_zero_one,
)


class ResnetBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator3D(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels=32, num_res_blocks=4):
        super().__init__()

        layers = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_channels, base_channels, kernel_size=7, bias=False),
            nn.InstanceNorm3d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        channels = base_channels
        for _ in range(2):
            next_channels = channels * 2
            layers.extend(
                [
                    nn.Conv3d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(next_channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels

        for _ in range(num_res_blocks):
            layers.append(ResnetBlock3D(channels))

        for _ in range(2):
            next_channels = channels // 2
            layers.extend(
                [
                    nn.ConvTranspose3d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(next_channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels

        layers.extend(
            [
                nn.ReflectionPad3d(3),
                nn.Conv3d(channels, output_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator3D(nn.Module):
    def __init__(self, input_channels, base_channels=32, n_layers=3):
        super().__init__()

        layers = [
            nn.Conv3d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for layer_idx in range(1, n_layers):
            next_channels = min(base_channels * (2 ** layer_idx), base_channels * 8)
            stride = 1 if layer_idx == n_layers - 1 else 2
            layers.extend(
                [
                    nn.Conv3d(
                        channels,
                        next_channels,
                        kernel_size=4,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(next_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels

        layers.append(nn.Conv3d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GANLoss:
    def __init__(self):
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size <= 0:
            return images.detach()

        selected_images = []
        for image in images:
            image = image.detach().unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                selected_images.append(image)
                continue

            if random.random() > 0.5:
                pool_idx = random.randint(0, self.pool_size - 1)
                cached_image = self.images[pool_idx].clone()
                self.images[pool_idx] = image.clone()
                selected_images.append(cached_image)
            else:
                selected_images.append(image)

        return torch.cat(selected_images, dim=0)


class CTtoPETCycleGAN:
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        base_channels=32,
        num_res_blocks=4,
        discriminator_channels=32,
        discriminator_layers=3,
        device="cuda",
    ):
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.discriminator_channels = discriminator_channels
        self.discriminator_layers = discriminator_layers

        self._build_networks()
        self.gan_loss = GANLoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()

    def _build_networks(self):
        self.generator_ab = ResnetGenerator3D(
            self.input_channels,
            self.output_channels,
            base_channels=self.base_channels,
            num_res_blocks=self.num_res_blocks,
        ).to(self.device)
        self.generator_ba = ResnetGenerator3D(
            self.output_channels,
            self.input_channels,
            base_channels=self.base_channels,
            num_res_blocks=self.num_res_blocks,
        ).to(self.device)
        self.discriminator_a = PatchDiscriminator3D(
            self.input_channels,
            base_channels=self.discriminator_channels,
            n_layers=self.discriminator_layers,
        ).to(self.device)
        self.discriminator_b = PatchDiscriminator3D(
            self.output_channels,
            base_channels=self.discriminator_channels,
            n_layers=self.discriminator_layers,
        ).to(self.device)

    @staticmethod
    def set_requires_grad(networks, requires_grad):
        for network in networks:
            for parameter in network.parameters():
                parameter.requires_grad = requires_grad

    @torch.no_grad()
    def generate(self, inputs):
        self.generator_ab.eval()
        return self.generator_ab(inputs)

    def save(self, path):
        torch.save(
            {
                "model_config": {
                    "input_channels": self.input_channels,
                    "output_channels": self.output_channels,
                    "base_channels": self.base_channels,
                    "num_res_blocks": self.num_res_blocks,
                    "discriminator_channels": self.discriminator_channels,
                    "discriminator_layers": self.discriminator_layers,
                },
                "generator_ab_state_dict": self.generator_ab.state_dict(),
                "generator_ba_state_dict": self.generator_ba.state_dict(),
                "discriminator_a_state_dict": self.discriminator_a.state_dict(),
                "discriminator_b_state_dict": self.discriminator_b.state_dict(),
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        model_config = checkpoint.get("model_config")
        if model_config is not None:
            needs_rebuild = any(
                [
                    model_config.get("input_channels", self.input_channels) != self.input_channels,
                    model_config.get("output_channels", self.output_channels) != self.output_channels,
                    model_config.get("base_channels", self.base_channels) != self.base_channels,
                    model_config.get("num_res_blocks", self.num_res_blocks) != self.num_res_blocks,
                    model_config.get("discriminator_channels", self.discriminator_channels)
                    != self.discriminator_channels,
                    model_config.get("discriminator_layers", self.discriminator_layers)
                    != self.discriminator_layers,
                ]
            )
            if needs_rebuild:
                self.input_channels = model_config.get("input_channels", self.input_channels)
                self.output_channels = model_config.get("output_channels", self.output_channels)
                self.base_channels = model_config.get("base_channels", self.base_channels)
                self.num_res_blocks = model_config.get("num_res_blocks", self.num_res_blocks)
                self.discriminator_channels = model_config.get(
                    "discriminator_channels", self.discriminator_channels
                )
                self.discriminator_layers = model_config.get(
                    "discriminator_layers", self.discriminator_layers
                )
                self._build_networks()

        if "generator_ab_state_dict" in checkpoint:
            self.generator_ab.load_state_dict(checkpoint["generator_ab_state_dict"])
            self.generator_ba.load_state_dict(checkpoint["generator_ba_state_dict"])
            self.discriminator_a.load_state_dict(checkpoint["discriminator_a_state_dict"])
            self.discriminator_b.load_state_dict(checkpoint["discriminator_b_state_dict"])
        else:
            self.generator_ab.load_state_dict(checkpoint)

        print(f"Model loaded from {path}")

def train_epoch(
    cyclegan,
    loader,
    optimizer_g,
    optimizer_d,
    input_key,
    target_key,
    device,
    epoch,
    epochs,
    use_fdg_condition=False,
    fdg_key="fdg_pt",
    lambda_cycle=10.0,
    lambda_identity=5.0,
    fake_a_pool=None,
    fake_b_pool=None,
):
    cyclegan.generator_ab.train()
    cyclegan.generator_ba.train()
    cyclegan.discriminator_a.train()
    cyclegan.discriminator_b.train()

    metrics = {
        "generator": [],
        "discriminator": [],
        "cycle": [],
        "identity": [],
        "gan_ab": [],
        "gan_ba": [],
    }

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for batch in progress:
        real_a, real_b = get_pair(
            batch,
            input_key,
            target_key,
            device,
            use_fdg_condition,
            fdg_key,
        )

        cyclegan.set_requires_grad(
            [cyclegan.discriminator_a, cyclegan.discriminator_b], False
        )
        optimizer_g.zero_grad(set_to_none=True)

        fake_b = cyclegan.generator_ab(real_a)
        fake_a = cyclegan.generator_ba(real_b)
        recon_a = cyclegan.generator_ba(fake_b)
        recon_b = cyclegan.generator_ab(fake_a)

        loss_gan_ab = cyclegan.gan_loss(cyclegan.discriminator_b(fake_b), True)
        loss_gan_ba = cyclegan.gan_loss(cyclegan.discriminator_a(fake_a), True)
        loss_cycle_a = cyclegan.cycle_loss(recon_a, real_a) * lambda_cycle
        loss_cycle_b = cyclegan.cycle_loss(recon_b, real_b) * lambda_cycle

        identity_active = lambda_identity > 0 and real_a.shape[1] == real_b.shape[1]
        loss_identity_a = torch.tensor(0.0, device=device)
        loss_identity_b = torch.tensor(0.0, device=device)
        if identity_active:
            identity_a = cyclegan.generator_ba(real_a)
            identity_b = cyclegan.generator_ab(real_b)
            loss_identity_a = cyclegan.identity_loss(identity_a, real_a) * lambda_identity
            loss_identity_b = cyclegan.identity_loss(identity_b, real_b) * lambda_identity

        total_cycle = loss_cycle_a + loss_cycle_b
        total_identity = loss_identity_a + loss_identity_b
        loss_g = loss_gan_ab + loss_gan_ba + total_cycle + total_identity
        loss_g.backward()
        optimizer_g.step()

        cyclegan.set_requires_grad(
            [cyclegan.discriminator_a, cyclegan.discriminator_b], True
        )
        optimizer_d.zero_grad(set_to_none=True)

        pooled_fake_a = fake_a_pool.query(fake_a) if fake_a_pool is not None else fake_a.detach()
        pooled_fake_b = fake_b_pool.query(fake_b) if fake_b_pool is not None else fake_b.detach()

        loss_d_a_real = cyclegan.gan_loss(cyclegan.discriminator_a(real_a), True)
        loss_d_a_fake = cyclegan.gan_loss(cyclegan.discriminator_a(pooled_fake_a), False)
        loss_d_a = 0.5 * (loss_d_a_real + loss_d_a_fake)

        loss_d_b_real = cyclegan.gan_loss(cyclegan.discriminator_b(real_b), True)
        loss_d_b_fake = cyclegan.gan_loss(cyclegan.discriminator_b(pooled_fake_b), False)
        loss_d_b = 0.5 * (loss_d_b_real + loss_d_b_fake)

        loss_d = loss_d_a + loss_d_b
        loss_d.backward()
        optimizer_d.step()

        metrics["generator"].append(loss_g.item())
        metrics["discriminator"].append(loss_d.item())
        metrics["cycle"].append(total_cycle.item())
        metrics["identity"].append(total_identity.item())
        metrics["gan_ab"].append(loss_gan_ab.item())
        metrics["gan_ba"].append(loss_gan_ba.item())

        progress.set_postfix(
            loss_g=f"{loss_g.item():.4f}",
            loss_d=f"{loss_d.item():.4f}",
            cycle=f"{total_cycle.item():.4f}",
        )

    return {key: float(np.mean(values)) for key, values in metrics.items()}


@torch.no_grad()
def run_inference(cyclegan, test_loader, args):
    progress = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="CycleGAN inference",
    )

    global_sample_idx = 0

    for batch_idx, batch in progress:
        batch_size = batch[args.target_key].shape[0]
        progress.set_postfix(batch=batch_idx, batch_size=batch_size)

        condition, target = get_pair(
            batch,
            args.input_key,
            args.target_key,
            args.device,
            args.use_fdg_condition,
            args.fdg_key,
        )

        prediction = cyclegan.generate(condition)
        prediction = map_minus_one_one_to_zero_one(prediction, clamp_output=True)
        target = map_minus_one_one_to_zero_one(target, clamp_output=True)

        print("generated sample min/max:", prediction.min().item(), prediction.max().item())

        for sample_idx in range(batch_size):
            sample_name = f"sample_{global_sample_idx:04d}"
            progress.set_postfix(sample=sample_name)

            case_dir = os.path.join(args.output_dir, sample_name)
            os.makedirs(case_dir, exist_ok=True)

            pred_i = prediction[sample_idx].unsqueeze(0)
            target_i = target[sample_idx].unsqueeze(0)

            sitk.WriteImage(
                tensor_to_itk(pred_i),
                os.path.join(case_dir, "psma_prediction.nii.gz"),
            )
            sitk.WriteImage(
                tensor_to_itk(target_i),
                os.path.join(case_dir, "psma_gt.nii.gz"),
            )

            global_sample_idx += 1
