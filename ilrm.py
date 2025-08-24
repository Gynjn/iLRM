import numpy as np
import torch
from torch import nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange
import traceback
from gsplat import rasterization
import torch.nn.functional as F
import os
import math
from transformer import SelfAttnBlock, ReadBlock
from utils import (
    compute_rays, 
    compute_rays_resolution, 
    compute_rays_resolution_offset
)
import time

def init_weights(m, std=0.02):
    """Initialize weights for linear and embedding layers.
    
    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Processor(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.model_arch=["r", "s"] * (config.model.transformer.n_layer // 2)
        self.blocks = nn.ModuleList()

        for i, arch in enumerate(self.model_arch):
            if arch == "r":
                self.blocks.append(ReadBlock(config.model.transformer.d, config.model.transformer.d_head))
            elif arch == "s":
                self.blocks.append(SelfAttnBlock(config.model.transformer.d, config.model.transformer.d_head))
            self.blocks[-1].apply(init_weights)

    def forward(
        self,
        viewpoint_tokens,
        input_tokens,
        V, use_checkpoint=True
    ):
        for i, arch in enumerate(self.model_arch):
            if use_checkpoint:
                if arch == "r":
                    viewpoint_tokens = torch.utils.checkpoint.checkpoint(
                        self.blocks[i], viewpoint_tokens, input_tokens, V,
                        use_reentrant=False,
                    )
                elif arch == "s":
                    viewpoint_tokens = torch.utils.checkpoint.checkpoint(
                        self.blocks[i], viewpoint_tokens,
                        use_reentrant=False
                    )
                else:
                    raise NotImplementedError

            else:
                if arch == "r":
                    viewpoint_tokens = self.blocks[i](
                        viewpoint_tokens, input_tokens, V
                    )
                elif arch == "s":
                    viewpoint_tokens = self.blocks[i](viewpoint_tokens)
                else:
                    raise NotImplementedError
                
        return viewpoint_tokens


class IterativeLRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pose_keys = ["ray_o", "ray_d", "o_cross_d"]
        self.posed_image_keys = self.pose_keys + ["normalized_image"]
        self._init_tokenizers()
        self.processor = Processor(config)
        self.vp_factor = config.model.viewpoint_factor

    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer
        self.viewpoint_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.viewpoint_tokenizer.in_channels,
            patch_size = self.config.model.viewpoint_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.viewpoint_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.viewpoint_tokenizer.patch_size**2) * \
                    (2 + 3 + (self.config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1),
                bias=False,
            ),
        )
        self.viewpoint_token_decoder.apply(init_weights)

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
            nn.LayerNorm(d_model, bias=False),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def render(self, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, W, H):
        B, V, _ = test_intr.shape
        renderings = []
        for i in range(B):
            xyz_i = xyz[i]
            feature_i = feature[i]
            scale_i = scale[i]
            scale_i = scale_i.exp()
            rotation_i = rotation[i]
            rotation_i = F.normalize(rotation_i, p=2, dim=-1)
            opacity_i = opacity[i]
            opacity_i = opacity_i.sigmoid().squeeze(-1)
            test_w2c_i = test_c2ws[i].float().inverse() # (V, 4, 4)
            test_intr_i = torch.zeros(V, 3, 3).to(test_intr.device)
            test_intr_i[:, 0, 0] = test_intr[i, :, 0]
            test_intr_i[:, 1, 1] = test_intr[i, :, 1]
            test_intr_i[:, 0, 2] = test_intr[i, :, 2]
            test_intr_i[:, 1, 2] = test_intr[i, :, 3]
            test_intr_i[:, 2, 2] = 1
            rendering, _, _ = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                                            test_w2c_i, test_intr_i, W, H, 
                                            packed=False,
                                            absgrad=False,
                                            sparse_grad=False,
                                            sh_degree=self.config.model.gaussians.sh_degree, 
                                            near_plane=self.config.model.gaussians.near_plane, far_plane=self.config.model.gaussians.far_plane,
                                            render_mode="RGB",
                                            backgrounds=torch.ones(V, 3).to(test_c2ws.device),
                                            rasterize_mode='classic') # (V, H, W, 3) 
            renderings.append(rendering)
        return torch.stack(renderings, dim=0) # (B, V, H, W, 3)   

    def render_one(self, xyz, feature, scale, rotation, opacity, test_c2w, test_intr, 
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        rotation = F.normalize(rotation, p=2, dim=-1)
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree, 
                                        near_plane=near_plane, far_plane=far_plane,
                                        packed=False,
                                        absgrad=False,
                                        sparse_grad=False,                                        
                                        render_mode="RGB",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 3) 
        return rendering # (1, H, W, 3)

    def forward(
        self,
        input_data_dict,
        target_data_dict,
        finetune=False,
        save_video=False,
    ):
        torch.cuda.synchronize()
        inference_start = time.time()        
        # Do not autocast during the data processing stage
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            b, v, _, h, w = input_data_dict["image"].size()
            h_factor = h // self.vp_factor
            w_factor = w // self.vp_factor

            i_fxfycxcy = input_data_dict["fxfycxcy"]
            i_c2w = input_data_dict["c2w"]
            t_fxfycxcy = target_data_dict["fxfycxcy"]
            t_c2w = target_data_dict["c2w"]

            i_ray_o, i_ray_d = compute_rays(i_fxfycxcy, i_c2w, h, w)
            i_o_cross_d = torch.cross(i_ray_o, i_ray_d, dim=2)
            i_normalized_image = input_data_dict["image"] * 2.0 - 1.0
            i_posed_images = torch.concat(
                [i_ray_o, i_ray_d, i_o_cross_d, i_normalized_image], dim=2
            )

            v_ray_o, v_ray_d = compute_rays_resolution(i_fxfycxcy, i_c2w, h, w, factor=self.vp_factor)
            v_o_cross_d = torch.cross(v_ray_o, v_ray_d, dim=2)
            v_viewpoints = torch.concat(
                [v_ray_o, v_ray_d, v_o_cross_d], dim=2
            )

        input_tokens = self.image_tokenizer(i_posed_images)
        viewpoint_tokens = self.viewpoint_tokenizer(v_viewpoints)
        output_tokens = self.processor(
            viewpoint_tokens,
            input_tokens,
            v
        )
        gaussians = self.viewpoint_token_decoder(output_tokens)        
        gaussians = rearrange(
            gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d", v=v, 
            hh=h_factor // self.config.model.viewpoint_tokenizer.patch_size, 
            ww=w_factor // self.config.model.viewpoint_tokenizer.patch_size, 
            ph=self.config.model.viewpoint_tokenizer.patch_size, 
            pw=self.config.model.viewpoint_tokenizer.patch_size)
        offset, xyz, feature, scale, rotation, opacity = torch.split(gaussians, [2, 3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=-1)
        offset = offset.float()
        xyz = xyz.float()
        feature = feature.float()
        scale = scale.float()
        rotation = rotation.float()
        opacity = opacity.float()
        with torch.autocast(device_type="cuda", enabled=False):
            offset_factor = rearrange(
                (offset.sigmoid() - 0.5), 
                "b (v h w) d -> (b v) (h w) d", v=v, h=h_factor, w=w_factor)            
            v_ray_o_gs, v_ray_d_gs = compute_rays_resolution_offset(
                i_fxfycxcy, i_c2w, h, w, offset_factor, factor=self.vp_factor
            )
            feature = feature.view(b, v*h_factor*w_factor, (self.config.model.gaussians.sh_degree + 1) ** 2, 3).contiguous()
            scale = (scale + self.config.model.gaussians.scale_bias).clamp(max = self.config.model.gaussians.scale_max) 
            opacity = opacity + self.config.model.gaussians.opacity_bias                
            dist = xyz.mean(dim=-1, keepdim=True).sigmoid() * self.config.model.gaussians.max_dist # (B, V*H*W, 1)
            xyz = dist * v_ray_d_gs + v_ray_o_gs

        gaussians = {
            "xyz": xyz,
            "feature": feature,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity
        }

        torch.cuda.synchronize()
        print(time.time() - inference_start, "inference_speed")

        if finetune:
            result = edict(
                input=input_data_dict,
                target=target_data_dict,
                render=None,
                gaussians=gaussians, 
                )
            return result            

        if save_video:
            input_intr = input_data_dict["fxfycxcy"][0]
            input_c2ws = input_data_dict["c2w"][0]
            gaussian_first = {k: v[0] for k, v in gaussians.items()}

            scene_name = input_data_dict["scene_name"][0][:5]
            output_dir = os.path.join(self.config.inference.out_dir, "videos", scene_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            self.save_input_video(
                input_intr, input_c2ws, gaussian_first, h, w,
                os.path.join(output_dir, "input_traj.mp4"),
                insert_frame_num=16
            )

        xyz = gaussians["xyz"]
        feature = gaussians["feature"]
        scale = gaussians["scale"]
        rotation = gaussians["rotation"]
        opacity = gaussians["opacity"]      

        # Avoid OOM by chunked rendering
        chunk_size = 10
        num_chunks = math.ceil(target_data_dict["image"].shape[1] / chunk_size)
        renderings = []

        with torch.autocast(device_type="cuda", enabled=False):
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, target_data_dict["c2w"].shape[1])
                t_c2w_chunk = t_c2w[:, start_idx:end_idx]
                t_fxfycxcy_chunk = t_fxfycxcy[:, start_idx:end_idx]

                rendering_curr = self.render(
                    xyz, 
                    feature, 
                    scale, 
                    rotation, 
                    opacity, 
                    t_c2w_chunk, 
                    t_fxfycxcy_chunk, 
                    w, h
                )
                renderings.append(rendering_curr)

        renderings = torch.cat(renderings, dim=1) # (B, V, H, W, 3)

        renderings = renderings.permute(0, 1, 4, 2, 3).contiguous() # (B, V, 3, H, W)

        result = edict(
            input=input_data_dict,
            target=target_data_dict,
            render=renderings,    
            )

        return result
    
    def save_input_video(self, input_intr, input_c2ws, gaussian_dict, H, W, save_path, insert_frame_num = 16):
        """
        Interpolate input frames and save rendered video
        input_intr: (V, 4), (fx, fy, cx, cy)
        input_c2ws: (V, 4, 4)
        """
        import cv2
        from camera_utils import get_interpolated_poses_many
        import subprocess
        V = input_intr.shape[0]
        device = input_intr.device
        input_intr = input_intr.detach().cpu().float()
        input_c2ws = input_c2ws.detach().cpu().float()

        input_intr_mat = torch.zeros((V, 3, 3))
        input_intr_mat[:, 0, 0] = input_intr[:, 0]
        input_intr_mat[:, 1, 1] = input_intr[:, 1]
        input_intr_mat[:, 0, 2] = input_intr[:, 2]
        input_intr_mat[:, 1, 2] = input_intr[:, 3]
        input_c2ws = torch.cat([input_c2ws, input_c2ws[:1]], dim=0) # wrap around
        input_intr_mat = torch.cat([input_intr_mat, input_intr_mat[:1]], dim=0) # wrap around
        c2ws, intr_mat, _ = get_interpolated_poses_many(input_c2ws[:, :3, :4], input_intr_mat, steps_per_transition = insert_frame_num)
        V = c2ws.shape[0]
        c2ws_mat = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        c2ws_mat[:, :3, :4] = c2ws
        intr_fxfycxcy = torch.zeros(V, 4)
        intr_fxfycxcy[:, 0] = intr_mat[:, 0, 0]
        intr_fxfycxcy[:, 1] = intr_mat[:, 1, 1]
        intr_fxfycxcy[:, 2] = intr_mat[:, 0, 2]
        intr_fxfycxcy[:, 3] = intr_mat[:, 1, 2]
        c2ws_mat = c2ws_mat.to(device)
        intr_fxfycxcy = intr_fxfycxcy.to(device)

        xyz = gaussian_dict["xyz"].detach().float().to(device) # (N, 3)
        feature = gaussian_dict["feature"].detach().float().to(device) # (N, (sh_degree+1)**2, 3)
        scale = gaussian_dict["scale"].detach().float().to(device) # (N, 3)
        rotation = gaussian_dict["rotation"].detach().float().to(device) # (N, 4)
        opacity = gaussian_dict["opacity"].detach().float().to(device).squeeze(-1) # (N, 1)

        renderings = []
        with torch.autocast(enabled=False, device_type="cuda"):
            for i in range(V):
                rendering = self.render_one(xyz, feature, scale, rotation, opacity, 
                                            c2ws_mat[i], intr_fxfycxcy[i], W, H, 
                                            self.config.model.gaussians.sh_degree, 
                                            self.config.model.gaussians.near_plane, 
                                            self.config.model.gaussians.far_plane)
                rendering = rendering.squeeze(0).clamp(0, 1).cpu().numpy() # (H, W, 3)
                rendering = (rendering * 255).astype(np.uint8)
                rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
                renderings.append(rendering)
        tmp_save_path = save_path.replace(".mp4", "_tmp.mp4")
        video_writer = cv2.VideoWriter(tmp_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
        for r in renderings:
            video_writer.write(r)
        video_writer.release()
        subprocess.run(f"ffmpeg -y -i {tmp_save_path} -vcodec libx264 -f mp4 {save_path} -loglevel quiet", shell=True) 
        os.remove(tmp_save_path)

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=True)
        return 0