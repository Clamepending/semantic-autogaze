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
    """Predicts per-patch similarity to a query embedding with spatial refinement."""

    def __init__(self, hidden_dim: int, embedding_dim: int, grid_size: int = 14,
                 num_frames: int = 16, use_spatial: bool = True):
        super().__init__()
        self.grid_size = grid_size
        self.num_frames = num_frames
        self.use_spatial = use_spatial

        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        # Deeper per-patch MLP
        self.patch_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if use_spatial:
            # Spatial conv refinement over the grid
            self.spatial_refine = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
            )

    def forward(self, patch_hidden_states: torch.Tensor, query_embedding: torch.Tensor):
        """
        Args:
            patch_hidden_states: (B, T*N_patches, hidden_dim) from AutoGaze decoder
            query_embedding: (B, embedding_dim) query embedding
        Returns:
            similarity_scores: (B, T*N_patches) predicted similarity to query
        """
        query_proj = self.query_proj(query_embedding)  # (B, hidden_dim)
        query_expanded = query_proj.unsqueeze(1).expand_as(patch_hidden_states)  # (B, N, hidden_dim)
        combined = torch.cat([patch_hidden_states, query_expanded], dim=-1)  # (B, N, hidden_dim*2)
        scores = self.patch_mlp(combined).squeeze(-1)  # (B, T*G*G)

        if self.use_spatial:
            B = scores.shape[0]
            T = scores.shape[1] // (self.grid_size * self.grid_size)
            G = self.grid_size
            # Reshape to per-frame grids: (B*T, 1, G, G)
            grids = scores.reshape(B * T, 1, G, G)
            refined = grids + self.spatial_refine(grids)  # residual connection
            scores = refined.reshape(B, T * G * G)

        return scores


class TeacherHead(nn.Module):
    """Teacher head that uses both AutoGaze hidden states AND CLIP visual features.

    During training, this head has access to rich CLIP visual patch features (768-dim)
    concatenated with AutoGaze hidden states (192-dim), giving it 960-dim input per patch.
    This allows it to learn a much better mapping than AutoGaze alone.

    The student head (SimilarityHead) then distills from this teacher, using only
    AutoGaze features at inference time.
    """

    def __init__(self, autogaze_dim: int, clip_visual_dim: int, text_dim: int,
                 grid_size: int = 14, num_frames: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self.num_frames = num_frames
        combined_dim = autogaze_dim + clip_visual_dim  # 192 + 768 = 960

        self.query_proj = nn.Linear(text_dim, combined_dim)
        self.patch_mlp = nn.Sequential(
            nn.Linear(combined_dim * 2, combined_dim),
            nn.GELU(),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Linear(combined_dim // 2, 1),
        )
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, autogaze_hidden: torch.Tensor, clip_visual: torch.Tensor,
                query_embedding: torch.Tensor):
        """
        Args:
            autogaze_hidden: (B, T*196, 192) AutoGaze decoder hidden states
            clip_visual: (B, T*196, 768) CLIP ViT-B/16 patch tokens
            query_embedding: (B, 512) CLIP text embedding
        Returns:
            scores: (B, T*196) logits
        """
        combined = torch.cat([autogaze_hidden, clip_visual], dim=-1)  # (B, N, 960)
        query_proj = self.query_proj(query_embedding)  # (B, 960)
        query_expanded = query_proj.unsqueeze(1).expand_as(combined)
        x = torch.cat([combined, query_expanded], dim=-1)  # (B, N, 1920)
        scores = self.patch_mlp(x).squeeze(-1)  # (B, T*196)

        B = scores.shape[0]
        T = scores.shape[1] // (self.grid_size * self.grid_size)
        G = self.grid_size
        grids = scores.reshape(B * T, 1, G, G)
        refined = grids + self.spatial_refine(grids)
        scores = refined.reshape(B, T * G * G)
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
