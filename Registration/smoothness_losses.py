import torch
import torch.nn as nn



def gradient_txyz(Txyz, fn):
    return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], axis=1)



def gradient_dx(arr):
    return (arr[:, 2:, 1:-1, 1:-1] - arr[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(arr):
    return (arr[:, 1:-1, 2:, 1:-1] - arr[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(arr):
    return (arr[:, 1:-1, 1:-1, 2:] - arr[:, 1:-1, 1:-1, :-2]) / 2

def l2_gradient(ddf, tensor_weights=None):
    dTdx = gradient_txyz(ddf, gradient_dx)
    dTdy = gradient_txyz(ddf, gradient_dy)
    dTdz = gradient_txyz(ddf, gradient_dz)
    L2 = dTdx**2 + dTdy** 2 + dTdz**2
    if tensor_weights is not None:
        return torch.mean(tensor_weights * L2)
    return torch.mean(L2)


def spatially_weighted_l2_gradient(flow, regularization_map):
    """
    Weighted local smoothness loss for a voxel-wise regularization map.

    Args:
        flow: Tensor with shape (B, 3, D, H, W).
        regularization_map: Tensor with shape (B, 1, D, H, W), values in (0, 1).
    """
    if regularization_map.dim() != 5 or regularization_map.shape[1] != 1:
        raise ValueError("regularization_map must have shape (B, 1, D, H, W).")

    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    w_x = regularization_map[:, :, :, 1:, :]
    w_y = regularization_map[:, :, 1:, :, :]
    w_z = regularization_map[:, :, :, :, 1:]

    return (
        torch.mean(w_x * dx.square())
        + torch.mean(w_y * dy.square())
        + torch.mean(w_z * dz.square())
    )


class BetaPriorLoss(nn.Module):
    """
    Beta prior negative log-likelihood for regularization maps.
    """

    def __init__(self, alpha=1.1, beta=1.0, eps=1e-6, mode="full_beta"):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.mode = mode

    def forward(self, regularization_map):
        w = torch.clamp(regularization_map, self.eps, 1.0 - self.eps)

        if self.mode == "repo_logbeta":
            return (1.0 - self.alpha) * torch.mean(torch.log(w))

        if self.mode != "full_beta":
            raise ValueError(f"Unsupported beta prior mode: {self.mode}")

        alpha = torch.as_tensor(self.alpha, dtype=w.dtype, device=w.device)
        beta = torch.as_tensor(self.beta, dtype=w.dtype, device=w.device)
        log_beta = (
            torch.lgamma(alpha)
            + torch.lgamma(beta)
            - torch.lgamma(alpha + beta)
        )
        log_prob = (
            (alpha - 1.0) * torch.log(w)
            + (beta - 1.0) * torch.log(1.0 - w)
            - log_beta
        )
        return -torch.mean(log_prob)
