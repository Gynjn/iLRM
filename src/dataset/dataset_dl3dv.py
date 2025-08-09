import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import numpy as np

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    augment: bool
    make_baseline_1: bool    
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    ori_image_shape: list[int]
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    shuffle_val: bool = True
    no_mix_test_set: bool = True
    min_views: int = 0
    max_views: int = 0
    sort_target_index: Optional[bool] = False
    sort_context_index: Optional[bool] = False
    use_index_to_load_chunk: Optional[bool] = False


class DatasetDL3DV(IterableDataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for i, root in enumerate(cfg.roots):
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            self.chunks.extend(root_chunks)
        if self.stage == "test":
            # fast testing
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
        if self.stage == "val":
            self.chunks = self.chunks * int(1e6 // len(self.chunks))

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            for run_idx in range(len(chunk)):
                example = chunk[run_idx]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])

                scene = example["key"]

                # try:
                out_data = self.view_sampler.sample(
                    scene,
                    extrinsics,
                    intrinsics,
                    min_context_views=self.cfg.min_views,
                    max_context_views=self.cfg.max_views,
                )
                if isinstance(out_data, tuple):
                    context_indices, target_indices = out_data[:2]
                    c_list = [
                        (
                            # context_indices.sort()[0]
                            # if self.cfg.sort_context_index
                            # else context_indices
                            context_indices
                        )
                    ]
                    t_list = [
                        (
                            # target_indices.sort()[0]
                            # if self.cfg.sort_target_index
                            # else target_indices
                            target_indices
                        )
                    ]
                if isinstance(out_data, list):
                    c_list = [
                        (
                            # a.context.sort()[0]
                            # if self.cfg.sort_context_index
                            # else a.context
                            a.context
                        )
                        for a in out_data
                    ]
                    t_list = [
                        (
                            # a.target.sort()[0]
                            # if self.cfg.sort_target_index
                            # else a.target
                            a.target
                        )
                        for a in out_data
                    ]


                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                for context_indices, target_indices in zip(c_list, t_list):
                    # Load the images.
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]

                    try:
                        context_images = self.convert_images(context_images)
                    except (OSError, TypeError):
                        # some data might be corrupted
                        continue

                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]

                    try:
                        target_images = self.convert_images(target_images)
                    except (OSError, TypeError):
                        # some data might be corrupted
                        continue

                    # Skip the example if the images don't have the right shape.
                    expected_shape = tuple(
                        [3, *self.cfg.ori_image_shape]
                    )  # (3, 270, 480) or (3, 540, 960)

                    context_image_invalid = context_images.shape[1:] != expected_shape
                    target_image_invalid = target_images.shape[1:] != expected_shape

                    if self.cfg.skip_bad_shape and (
                        context_image_invalid or target_image_invalid
                    ):
                        print(
                            f"Skipped bad example {example['key']}. Context shape was "
                            f"{context_images.shape}, target shape was "
                            f"{target_images.shape}, "
                            f"and expected shape was {expected_shape}"
                        )
                        continue

                    # check the extrinsics
                    if any(torch.isnan(torch.det(extrinsics[context_indices][:, :3, :3]))):
                        # print('invalid extrinsics')
                        continue

                    if any(torch.isnan(torch.det(extrinsics[target_indices][:, :3, :3]))):
                        # print('invalid extrinsics')
                        continue

                    # check the extrinsics: translation could be very large
                    # https://github.com/DL3DV-10K/Dataset/issues/34
                    if (extrinsics[context_indices][:, :3, 3] > 1e3).any():
                        # print('extremely large camera translation')
                        continue

                    if (extrinsics[target_indices][:, :3, 3] > 1e3).any():
                        # print('extremely large camera translation')
                        continue

                    if not torch.allclose(torch.det(extrinsics[context_indices][:, :3, :3]), torch.det(extrinsics[context_indices][:, :3, :3]).new_tensor(1)):
                        # print('invalid extrinsics')
                        continue
                    if not torch.allclose(torch.det(extrinsics[target_indices][:, :3, :3]), torch.det(extrinsics[target_indices][:, :3, :3]).new_tensor(1)):
                        # print('invalid extrinsics')
                        continue
                    
                                    
                    context_extrinsics = extrinsics[context_indices]

                    position_avg = context_extrinsics[:, :3, 3].mean(dim=0)
                    forward_avg = context_extrinsics[:, :3, 2].mean(dim=0)
                    down_avg = context_extrinsics[:, :3, 1].mean(dim=0)

                    forward_avg = F.normalize(forward_avg, dim=0)
                    down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
                    right_avg = torch.cross(down_avg, forward_avg, dim=0)
                    pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1)
                    pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=pos_avg.device)], dim=0)
                    pos_avg_inv = torch.inverse(pos_avg)

                    position_max = context_extrinsics[:, :3, 3].abs().max()
                    scene_scale = 1.0 / position_max

                    extrinsics = torch.matmul(pos_avg_inv.unsqueeze(0), extrinsics)
                    extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * scene_scale.reshape(1, 1)
                    example_out = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "index": target_indices,
                        },         
                        "scene": scene,
                    }

                    if self.stage == "train" and self.cfg.augment:
                        example_out = apply_augmentation_shim(example_out)
                    yield example_out

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                if not (root / data_stage).is_dir():
                    continue

                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v)
                         for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if self.stage in ['train', 'test']:
            return (
                min(
                    len(self.index.keys()) * self.cfg.test_times_per_scene,
                    self.cfg.test_len,
                )
                if self.stage == "test" and self.cfg.test_len > 0
                else len(self.index.keys()) * self.cfg.train_times_per_scene
            )
        else:
            # set a very large value here to ensure the validation keep going
            # and do not exhaust; it will be wrap to length 1 anyway.
            return int(1e10)
