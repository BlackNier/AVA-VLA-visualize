from typing import Dict, Optional

import torch

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class TemporalFeatureExtractor:
    def __init__(self):
        self.supported_sources = {"vision", "action"}

    def _validate_inputs(
        self,
        temporal_feature_source: str,
        last_hidden_states: torch.Tensor,
        frame_batch: Dict[str, torch.Tensor],
        num_patches: int,
    ) -> None:
        assert temporal_feature_source in self.supported_sources
        assert last_hidden_states is not None and last_hidden_states.dim() == 3
        assert "input_ids" in frame_batch

        batch_size = last_hidden_states.shape[0]
        assert frame_batch["input_ids"].shape[0] == batch_size
        if num_patches is not None:
            assert (num_patches + frame_batch["input_ids"].shape[1]) == last_hidden_states.shape[1]

    def extract_temporal_features(
        self,
        temporal_feature_source: str,
        last_hidden_states: torch.Tensor,
        frame_batch: Dict[str, torch.Tensor],
        vla_model,
        num_patches: Optional[int] = None,
    ) -> torch.Tensor:
        self._validate_inputs(temporal_feature_source, last_hidden_states, frame_batch, num_patches)

        if temporal_feature_source is None:
            return None

        if temporal_feature_source == "vision":
            return self._extract_from_vision_patches(last_hidden_states, vla_model)
        elif temporal_feature_source == "action":
            return self._extract_from_action_patches(last_hidden_states, frame_batch, num_patches)

    def _extract_from_vision_patches(self, last_hidden_states: torch.Tensor, vla_model) -> torch.Tensor:
        if hasattr(vla_model, "module"):
            vision_backbone = vla_model.module.vision_backbone
        else:
            vision_backbone = vla_model.vision_backbone

        pure_vision_patches = vision_backbone.get_num_patches() * vision_backbone.get_num_images_in_input()

        temporal_features = last_hidden_states[:, 1 : pure_vision_patches + 1]
        return temporal_features

    def _extract_from_action_patches(
        self,
        last_hidden_states: torch.Tensor,
        frame_batch,
        num_patches,
    ) -> torch.Tensor:
        batch_size = last_hidden_states.shape[0]

        text_input_ids = frame_batch["labels"][:, 1:]
        current_action_mask = get_current_action_mask(text_input_ids)
        next_actions_mask = get_next_actions_mask(text_input_ids)

        text_hidden_states = last_hidden_states[:, num_patches:-1]
        temporal_features = text_hidden_states[current_action_mask | next_actions_mask].reshape(
            batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
        )

        return temporal_features
