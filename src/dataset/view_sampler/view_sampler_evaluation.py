import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...evaluation.evaluation_index_generator import IndexEntry
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler


def insert_between_indices(context_indices):
    between_indices = context_indices[:-1] + (context_indices[1:] - context_indices[:-1]) // 2
    interleaved = torch.empty(context_indices.shape[0] + between_indices.shape[0], dtype=torch.int64, device=context_indices.device)
    interleaved[0::2] = context_indices
    interleaved[1::2] = between_indices

    return interleaved

@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int

class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:       
        entry = self.index.get(scene)
        num_views, _, _ = extrinsics.shape
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)

        # context_indices = insert_between_indices(context_indices)

        # for i in range(len(context_indices)):
        #     while context_indices[i] in target_indices:
        #         context_indices[i] = torch.clamp(context_indices[i] + 1, 0, num_views - 1)
        #         if context_indices[i] in target_indices:
        #             context_indices[i] = torch.clamp(context_indices[i] - 2, 0, num_views - 1)

        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
