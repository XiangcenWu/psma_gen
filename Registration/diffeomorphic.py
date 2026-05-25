import torch
import torch.nn.functional as F


def get_diffeomorphic_tag(enabled, velocity_scale=0.5, int_steps=7):
    if not enabled:
        return ""
    return f"_diffeomorphic_s{velocity_scale:g}_i{int_steps}"


def _make_sampling_grid(identity_grid, ddf):
    """
    identity_grid:
        shape (B, ndim, *spatial_size)
        normalized grid coordinates, usually in [-1, 1]

    ddf:
        shape (B, ndim, *spatial_size)
        normalized displacement field

    return:
        grid:
            shape (B, *spatial_size, ndim)
            ready for torch.nn.functional.grid_sample
    """
    grid = identity_grid.to(device=ddf.device, dtype=ddf.dtype) + ddf
    grid = torch.movedim(grid, 1, -1).contiguous()
    return grid


def compose_ddf(ddf_a, ddf_b, identity_grid, mode="bilinear", padding_mode="border", align_corners=True):
    """
    Compose two displacement fields.

    Given:
        phi_a(x) = x + ddf_a(x)
        phi_b(x) = x + ddf_b(x)

    Return:
        phi_ab = phi_b o phi_a

    In displacement form:
        ddf_ab(x) = ddf_a(x) + ddf_b(x + ddf_a(x))

    Args:
        ddf_a:
            shape (B, ndim, *spatial_size)

        ddf_b:
            shape (B, ndim, *spatial_size)

        identity_grid:
            shape (B, ndim, *spatial_size)

    Returns:
        composed_ddf:
            shape (B, ndim, *spatial_size)
    """
    grid_a = _make_sampling_grid(identity_grid, ddf_a)

    warped_ddf_b = F.grid_sample(
        ddf_b,
        grid_a,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    composed_ddf = ddf_a + warped_ddf_b

    return composed_ddf


def integrate_svf_scaling_and_squaring(
    velocity,
    identity_grid,
    int_steps=7,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
):
    """
    Integrate a stationary velocity field into a displacement field.

    Args:
        velocity:
            shape (B, ndim, *spatial_size)
            This is SVF, not final DDF.
            It should be in normalized grid coordinates.

        identity_grid:
            shape (B, ndim, *spatial_size)
            Normalized identity grid.

        int_steps:
            Number of scaling-and-squaring steps.
            Typical values: 5, 6, 7.

    Returns:
        ddf:
            shape (B, ndim, *spatial_size)
            Integrated displacement field in normalized coordinates.
    """
    if int_steps < 0:
        raise ValueError(f"int_steps must be >= 0, got {int_steps}")

    identity_grid = identity_grid.to(device=velocity.device, dtype=velocity.dtype)

    # scaling step
    ddf = velocity / (2 ** int_steps)

    # squaring step
    # phi <- phi o phi
    # ddf <- ddf + ddf o (Id + ddf)
    for _ in range(int_steps):
        ddf = compose_ddf(
            ddf_a=ddf,
            ddf_b=ddf,
            identity_grid=identity_grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    return ddf


def predict_diffeomorphic_ddf_and_grid(
    model,
    model_input,
    identity_grid,
    apply_tanh=True,
    velocity_scale=1.0,
    int_steps=7,
    return_velocity=False,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
):
    """
    Run a registration model and build a diffeomorphic grid_sample sampling grid.

    Compared with ordinary DDF prediction:

        raw_output = model(model_input)
        ddf = tanh(raw_output)
        grid = identity + ddf

    This function does:

        raw_output = model(model_input)
        velocity = tanh(raw_output) * velocity_scale
        ddf = integrate_svf(velocity)
        grid = identity + ddf

    Args:
        model:
            Registration network.

        model_input:
            Input to the model.

        identity_grid:
            shape (B, ndim, *spatial_size)
            Normalized identity grid.
            Same convention as your current DDF code.

        apply_tanh:
            If True, apply tanh to model output before treating it as velocity.

        velocity_scale:
            Scale applied to the SVF.
            Since your ddf/grid are in normalized coordinates, this should also be
            in normalized-coordinate units.

            Common starting values:
                0.1, 0.2, 0.5, 1.0

        int_steps:
            Scaling-and-squaring integration steps.
            Common values:
                5, 6, 7

        return_velocity:
            If True, also return the SVF velocity field.

    Returns:
        ddf:
            shape (B, ndim, *spatial_size)
            Integrated displacement field in normalized grid coordinates.

        grid:
            shape (B, *spatial_size, ndim)
            Ready for torch.nn.functional.grid_sample.

        velocity:
            Optional.
            shape (B, ndim, *spatial_size)
            The stationary velocity field before integration.
    """
    raw_output = model(model_input)

    # Interpret model output as SVF, not DDF.
    velocity = raw_output

    if apply_tanh:
        velocity = torch.tanh(velocity)

    velocity = velocity * velocity_scale

    # SVF -> DDF
    ddf = integrate_svf_scaling_and_squaring(
        velocity=velocity,
        identity_grid=identity_grid,
        int_steps=int_steps,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # DDF -> grid_sample grid
    grid = _make_sampling_grid(identity_grid, ddf)

    if return_velocity:
        return ddf, grid, velocity

    return ddf, grid
