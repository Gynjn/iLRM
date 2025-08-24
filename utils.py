import torch
from einops import rearrange

def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d

def compute_rays_resolution(fxfycxcy, c2w, h, w, factor=2):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    w_factor = w // factor
    h_factor = h // factor
    fxfycxcy_factor = fxfycxcy.clone() / factor

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w_factor, device=c2w.device)[None, :].expand(h_factor, -1)
    idx_y = torch.arange(h_factor, device=c2w.device)[:, None].expand(-1, w_factor)

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)
    idx_y = idx_y.flatten().expand(b * v, -1)

    fxfycxcy_factor = fxfycxcy_factor.reshape(b * v, 4)
    c2w = c2w.reshape(b * v, 4, 4)

    x = (idx_x + 0.5 - fxfycxcy_factor[:, 2:3]) / fxfycxcy_factor[:, 0:1]
    y = (idx_y + 0.5 - fxfycxcy_factor[:, 3:4]) / fxfycxcy_factor[:, 1:2]
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=1)
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h_factor*w_factor)

    ray_o = ray_o.reshape(b, v, 3, h_factor, w_factor)
    ray_d = ray_d.reshape(b, v, 3, h_factor, w_factor)

    return ray_o, ray_d

def compute_rays_resolution_offset(fxfycxcy, c2w, h, w, offset, factor=2):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    w_factor = w // factor
    h_factor = h // factor
    fxfycxcy_factor = fxfycxcy.clone() / factor

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w_factor, device=c2w.device)[None, :].expand(h_factor, -1)
    idx_y = torch.arange(h_factor, device=c2w.device)[:, None].expand(-1, w_factor)

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)
    idx_y = idx_y.flatten().expand(b * v, -1)

    fxfycxcy_factor = fxfycxcy_factor.reshape(b * v, 4)
    c2w = c2w.reshape(b * v, 4, 4)

    x = (idx_x + 0.5 + offset[..., 0] - fxfycxcy_factor[:, 2:3]) / fxfycxcy_factor[:, 0:1]
    y = (idx_y + 0.5 + offset[..., 1] - fxfycxcy_factor[:, 3:4]) / fxfycxcy_factor[:, 1:2]
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=1)
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h_factor*w_factor)

    ray_o = ray_o.reshape(b, v, 3, h_factor, w_factor)
    ray_d = ray_d.reshape(b, v, 3, h_factor, w_factor)

    ray_o = rearrange(ray_o, 'b v c h w -> b (v h w) c')
    ray_d = rearrange(ray_d, 'b v c h w -> b (v h w) c')

    return ray_o, ray_d