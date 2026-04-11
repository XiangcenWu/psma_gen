import torch
import torch.nn.functional as F

def map_zero_one_to_minus_one_one(image):
    return image * 2.0 - 1.0


def map_minus_one_one_to_zero_one(image, clamp_output=False):
    image = (image + 1.0) / 2.0
    if clamp_output:
        image = image.clamp_(0.0, 1.0)
    return image

def resize_half(x, size=(64, 64, 192), mode="trilinear"):
    return F.interpolate(x, size=size, mode=mode, align_corners=False)

def get_pair(
    batch,
    input_key,
    target_key,
    device,
    use_fdg_condition=False,
    fdg_key="fdg_pt",
):
    condition = batch[input_key].float().to(device)
    target = batch[target_key].float().to(device)

    # 下采样到一半分辨率
    condition = resize_half(condition)
    target = resize_half(target)

    if use_fdg_condition:
        fdg = batch[fdg_key].float().to(device)
        fdg = resize_half(fdg)
        condition = torch.cat([condition, fdg], dim=1)

    condition = map_zero_one_to_minus_one_one(condition)
    target = map_zero_one_to_minus_one_one(target)
    print(f"Condition shape: {condition.shape}, Target shape: {target.shape}")
    return condition, target

