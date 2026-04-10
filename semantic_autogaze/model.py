"""
Semantic AutoGaze: AutoGaze + SigLIP semantic similarity head.

Freezes pretrained AutoGaze weights and adds:
- An embedding projection layer (SigLIP dim → hidden dim)
- A per-patch similarity prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from autogaze.models.autogaze import AutoGaze, AutoGazeConfig
from autogaze.models.autogaze.modeling_autogaze import AutoGazeModel


class SimilarityHead(nn.Module):
    """Predicts per-patch similarity to a query embedding."""

    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_hidden_states: torch.Tensor, query_embedding: torch.Tensor):
        """
        Args:
            patch_hidden_states: (B, N_patches, hidden_dim) from AutoGaze decoder
            query_embedding: (B, embedding_dim) SigLIP embedding of query
        Returns:
            similarity_scores: (B, N_patches) predicted similarity to query
        """
        query_proj = self.query_proj(query_embedding)  # (B, hidden_dim)
        query_expanded = query_proj.unsqueeze(1).expand_as(patch_hidden_states)  # (B, N, hidden_dim)
        combined = torch.cat([patch_hidden_states, query_expanded], dim=-1)  # (B, N, hidden_dim*2)
        scores = self.head(combined).squeeze(-1)  # (B, N)
        return scores


class SemanticAutoGaze(nn.Module):
    """
    AutoGaze with a frozen backbone and a trainable semantic similarity head.

    Given a video and a query embedding (e.g., SigLIP embedding of a text prompt
    or a sampled patch), predicts per-patch similarity scores alongside the
    standard AutoGaze patch selection.
    """

    def __init__(
        self,
        autogaze_model: AutoGaze,
        embedding_dim: int = 768,  # SigLIP-base hidden dim
    ):
        super().__init__()

        self.autogaze = autogaze_model
        hidden_dim = autogaze_model.config.gaze_model_config.gaze_decoder_config.hidden_size
        # The encoder grid produces (input_img_size // kernel_size)^2 patches per frame
        grid = autogaze_model.config.gaze_model_config.input_img_size // autogaze_model.config.gaze_model_config.vision_model_config.kernel_size
        self.num_patches_per_frame = grid * grid  # 196 for 224/16
        self.patch_grid_size = grid  # 14 for 224/16
        self.num_vision_tokens = autogaze_model.num_vision_tokens_each_frame  # 265 multi-scale

        # Freeze all AutoGaze parameters
        for param in self.autogaze.parameters():
            param.requires_grad = False

        # Trainable similarity head
        self.similarity_head = SimilarityHead(hidden_dim, embedding_dim)

    def get_patch_hidden_states(self, video):
        """
        Run frozen AutoGaze encoder + decoder to extract per-patch hidden states.

        Args:
            video: (B, T, C, H, W)
        Returns:
            hidden_states: (B, T*N_patches, hidden_dim)
        """
        B, T = video.shape[:2]
        gaze_model = self.autogaze.gazing_model

        # Resize to AutoGaze input size
        video_resized = rearrange(video, 'b t c h w -> (b t) c h w')
        video_resized = F.interpolate(
            video_resized,
            size=(gaze_model.input_img_size, gaze_model.input_img_size),
            mode="bicubic", align_corners=False,
        )
        video_resized = rearrange(video_resized, '(b t) c h w -> b t c h w', b=B)

        # Run vision encoder + connector
        vision_features, _ = gaze_model.vision_model(video_resized)
        vision_features = vision_features.transpose(1, 2)
        vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
        vision_features = gaze_model.connector(vision_features)
        # vision_features: (B, T, N_patches_per_frame, hidden_dim)

        # Run through the LLaMA decoder to get contextualized hidden states
        # Flatten frames into sequence
        B, T_sub, N, C = vision_features.shape
        inputs_embeds = vision_features.reshape(B, T_sub * N, C)
        attention_mask = torch.ones(B, T_sub * N, device=video.device, dtype=torch.long)

        with torch.no_grad():
            decoder_outputs = gaze_model.gaze_decoder.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=attention_mask.cumsum(dim=-1) - 1,
            )

        hidden_states = decoder_outputs.last_hidden_state  # (B, T*N, hidden_dim)
        return hidden_states

    def forward(self, video, query_embedding, return_gaze=False, gazing_ratio=0.75, task_loss_requirement=0.7):
        """
        Args:
            video: (B, T, C, H, W)
            query_embedding: (B, embedding_dim) - SigLIP embedding to compute similarity against
            return_gaze: if True, also run AutoGaze generation to get gaze outputs
            gazing_ratio: max gazing ratio for AutoGaze
            task_loss_requirement: reconstruction quality threshold for AutoGaze
        Returns:
            dict with:
                - similarity_scores: (B, T*N_patches) predicted per-patch similarity
                - gaze_outputs: (optional) AutoGaze outputs if return_gaze=True
        """
        with torch.no_grad():
            hidden_states = self.get_patch_hidden_states(video)

        similarity_scores = self.similarity_head(hidden_states, query_embedding)

        result = {"similarity_scores": similarity_scores}

        if return_gaze:
            with torch.no_grad():
                gaze_outputs = self.autogaze(
                    {"video": video},
                    gazing_ratio=gazing_ratio,
                    task_loss_requirement=task_loss_requirement,
                    generate_only=True,
                )
            result["gaze_outputs"] = gaze_outputs

        return result

    def trainable_parameters(self):
        """Returns only the trainable parameters (similarity head)."""
        return self.similarity_head.parameters()
