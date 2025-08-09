from dataclasses import dataclass
from typing import Literal, Optional
from einops import rearrange

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ..types import Gaussians
from .encoder import Encoder
from .scalable.scaler import Scaler


@dataclass
class EncoderScaleCfg:
    name: Literal["scale"]
    patch_size: int
    dim: int
    head_dim: int
    num_layers: int


class EncoderScale(Encoder[EncoderScaleCfg]):

    def __init__(self, cfg: EncoderScaleCfg) -> None:
        super().__init__(cfg)

        self.scaler = Scaler(
            patch_size=cfg.patch_size,
            dim=cfg.dim,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        _, _, _, h, w = context["image"].shape

        gaussians = self.scaler.forward(
            context,
        )
        xyz_world = rearrange(
            gaussians["xyz_world"], 
            "b v (h w) srf -> b v (h w) () () srf", 
            h=h, w=w)
        densities = rearrange(
            (gaussians["opacity"]-2.0),
            "b v (h w) srf -> b v (h w) () srf",
            h=h,
            w=w,
        ).float()

        scale = (gaussians["scale"] - 6.9).clamp(max=-1.2).float()
        rotation = gaussians["rotation"].float()
        feature = rearrange(gaussians["feature"].float(), "... (xyz d_sh) -> ... xyz d_sh", xyz=3)

        return Gaussians(
            rearrange(
                xyz_world,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                rotation,
                "b v r wxyz -> b (v r) wxyz",
            ),
            rearrange(
                scale,
                "b v r xyz -> b (v r) xyz",
            ),
            rearrange(
                feature,
                "b v r c d_sh -> b (v r) c d_sh",
            ),
            rearrange(
                densities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.patch_size * 8, # 64 
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
