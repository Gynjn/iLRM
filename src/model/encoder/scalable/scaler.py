import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .transformer import ReadBlock, SelfAttnBlock


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Processor(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        head_dim,
        ):
        super().__init__()
        self.model_arch = ["r", "s"] * (num_layers // 2)
        self.blocks = nn.ModuleList()

        for i, arch in enumerate(self.model_arch):
            if arch == "r":
                self.blocks.append(ReadBlock(dim, head_dim))
            elif arch == "s":
                self.blocks.append(SelfAttnBlock(dim, head_dim))
            self.blocks[-1].apply(_init_weights)


    def forward(self, input, support, V,
                use_checkpoint=True):
        B = input.shape[0]
        for i, arch in enumerate(self.model_arch):
            if use_checkpoint:
                if arch == "r":
                    input = torch.utils.checkpoint.checkpoint(
                        self.blocks[i], input, support, V,
                        use_reentrant=False)
                elif arch == "s":
                    input = torch.utils.checkpoint.checkpoint(
                        self.blocks[i], input,
                        use_reentrant=False)
                else:
                    raise NotImplementedError
            else:
                if arch == "r":
                    input = self.blocks[i](
                        input, support, V)
                elif arch == "s":
                    input = self.blocks[i](
                        input)
                else:
                    raise NotImplementedError
        return input
    

class Scaler(nn.Module):
    def __init__(
        self,
        patch_size,
        dim,
        head_dim,
        num_layers,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim_start = dim
        self.dim_end = dim
        input_dim = 6
        support_dim = 9
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.input_tokenizer = nn.Linear(input_dim * self.patch_size ** 2, self.dim_start, bias=False)
        self.input_tokenizer.apply(_init_weights)
        self.input_layernorm = nn.LayerNorm(self.dim_start, bias=False)

        self.support_tokenizer = nn.Linear(support_dim * self.patch_size ** 2, self.dim_start, bias=False)
        self.support_tokenizer.apply(_init_weights)
        self.support_layernorm = nn.LayerNorm(self.dim_start, bias=False)

        self.processor = Processor(
            num_layers,
            dim,
            head_dim,
        )
        # we only utilize 0 degree of sh coefficients
        self.token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim_end, bias=False),
            nn.Linear(
                self.dim_end,
                (3 + (1) ** 2 * 3 + 3 + 4 + 1) * self.patch_size ** 2,
                bias=False,
            )
        )
        self.token_decoder.apply(_init_weights)

    def forward(self, 
                context,
                use_checkpoint=True,
               ):
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            input_images = context['image']
            B, V, _, H, W = input_images.shape

            input_c2ws = context['extrinsics']
            
            input_intr = context['intrinsics'].clone().detach()
            input_intr[:, :, 0] *= float(W)
            input_intr[:, :, 1] *= float(H)


            patch_size = self.patch_size
            hh = H // patch_size
            ww = W // patch_size

            ray_o = input_c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1).float() # (B, V, H*W, 3) # camera origin
            x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            x = (x.to(input_c2ws.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device).contiguous()
            y = (y.to(input_c2ws.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device).contiguous()
            x = (x - input_intr[:, :, 0:1, 2]) / input_intr[:, :, 0:1, 0] # (x - cx) / fx
            y = (y - input_intr[:, :, 1:2, 2]) / input_intr[:, :, 1:2, 1] # (y - cy) / fy
            ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (B, V, H*W, 3)
            ray_d = F.normalize(ray_d, p=2, dim=-1)
            ray_d = ray_d @ input_c2ws[:, :, :3, :3].transpose(-1, -2).contiguous() # (B, V, H*W, 3)

            input_image_cam = torch.cat([torch.cross(ray_o, ray_d, dim=-1),
                                        ray_d], dim=-1) # (B, V, H*W, 9)               


            support_image_cam = torch.cat([input_images.view(B, V, 3, -1).permute(0, 1, 3, 2).contiguous() * 2 - 1, 
                                torch.cross(ray_o, ray_d, dim=-1), ray_d], dim=-1)

        input_image_cam = rearrange(input_image_cam, 
                                    "b v (hh ph ww pw) d -> b (v hh ww) (ph pw d)", 
                                    hh=hh, ww=ww, 
                                    ph=patch_size, pw=patch_size)
        support_image_cam = rearrange(support_image_cam, 
                                      "b s (hh ph ww pw) d -> b (s hh ww) (ph pw d)", 
                                      hh=hh, ww=ww, 
                                      ph=patch_size, pw=patch_size)
        input_tokens = self.input_layernorm(
            self.input_tokenizer(input_image_cam)
        )
        support_tokens = self.support_layernorm(
            self.support_tokenizer(support_image_cam)
        )
        output_tokens = self.processor(
            input_tokens, support_tokens, V, 
            use_checkpoint=use_checkpoint
        )
        gaussians = self.token_decoder(output_tokens)
        gaussians = rearrange(
            gaussians, "b (v hh ww) (ph pw d) -> b v (hh ph ww pw) d",
            v=V, hh=hh, ww=ww, ph=patch_size, pw=patch_size
        )
        xyz, feature, scale, rotation, opacity = torch.split(
            gaussians, [3, (1)**2 *3, 3, 4, 1], dim=-1
        )
        depths = xyz.mean(dim=-1, keepdim=True).sigmoid() * 500.0

        xyz_world = depths * ray_d + ray_o
        return {
            "xyz_world": xyz_world, # b v (h w) 3
            "feature": feature, # b v (h w) (4+1)^2 * 3
            "scale": scale, # b v (h w) 3
            "rotation": rotation, # b v (h w) 4
            "opacity": opacity, # b v (h w) 1
        }