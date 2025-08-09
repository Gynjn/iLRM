from math import isqrt
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from gsplat.rendering import rasterization
import torch.nn.functional as F

def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    use_sh: bool = True,
) -> Float[Tensor, "batch view 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _, _ = extrinsics.shape
    h, w = image_shape

    Ks = intrinsics.clone().detach()
    Ks[:, :, 0] *= w
    Ks[:, :, 1] *= h

    w2c = extrinsics.float().inverse()
    gaussian_scales = torch.exp(gaussian_scales)
    gaussian_rotations = F.normalize(gaussian_rotations, p=2, dim=-1)
    gaussian_opacities = gaussian_opacities.sigmoid()
    all_images = []
    for i in range(b):
        render_colors, _, _ = rasterization(
            means=gaussian_means[i],
            scales=gaussian_scales[i],
            quats=gaussian_rotations[i],
            opacities=gaussian_opacities[i],
            colors=shs[i],
            viewmats=w2c[i],
            Ks=Ks[i],
            width=w,
            height=h,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode='classic',
            backgrounds=torch.ones(w2c[i].shape[0], 3, dtype=torch.float32).to(w2c[i].device),
            near_plane=0.01,
            far_plane=500.0,
            render_mode="RGB",
            sh_degree=degree,
        )
        image = render_colors[..., :3].permute(0, 3, 1, 2)
        all_images.append(image)
    return torch.stack(all_images)
