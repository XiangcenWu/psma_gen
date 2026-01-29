import torch



def gradient_txyz(Txyz, fn):
    return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], axis=1)



def gradient_dx(arr):
    return (arr[:, 2:, 1:-1, 1:-1] - arr[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(arr):
    return (arr[:, 1:-1, 2:, 1:-1] - arr[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(arr):
    return (arr[:, 1:-1, 1:-1, 2:] - arr[:, 1:-1, 1:-1, :-2]) / 2

def l2_gradient(ddf):
    dTdx = gradient_txyz(ddf, gradient_dx)
    dTdy = gradient_txyz(ddf, gradient_dy)
    dTdz = gradient_txyz(ddf, gradient_dz)
    return torch.mean(dTdx**2 + dTdy** 2 + dTdz**2)
