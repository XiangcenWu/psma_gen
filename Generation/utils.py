import torch


def map_zero_one_to_minus_one_one(image):
    return image * 2.0 - 1.0


def map_minus_one_one_to_zero_one(image, clamp_output=False):
    image = (image + 1.0) / 2.0
    if clamp_output:
        image = image.clamp_(0.0, 1.0)
    return image


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
