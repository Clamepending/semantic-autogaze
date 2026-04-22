"""
SemanticAutoGaze: Wrapper combining AutoGaze with text-conditioned semantic filtering.

Extends the AutoGaze pipeline with a lightweight semantic filtering head that
further prunes gazed patches based on text query relevance. The pipeline is:

  1. AutoGaze generates gazing_info (reconstruction-based patch selection)
  2. AutoGaze's decoder hidden states are extracted
  3. Semantic head scores each patch against the text query
  4. gazing_info is refined: only semantically relevant patches are kept

This produces a modified gazing_info dict that is directly compatible with
SigLIP's mask_with_gazing() and the rest of the NVILA pipeline.

Usage:
    from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper

    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name="nvidia/AutoGaze",
        head_ckpt="results/distill_bighead/best_bighead_student.pt",
        device="cuda:0",
    )

    # Full pipeline: video + text → filtered gazing_info
    gazing_info = wrapper.forward(
        video,                   # (B, T, C, H, W) preprocessed
        text_embedding,          # (B, 512) CLIP text embedding
        gazing_ratio=0.75,       # AutoGaze budget
        semantic_keep_ratio=0.5, # Fraction of gazed patches to keep
    )

    # Then pass to SigLIP:
    # siglip_outputs = siglip_model(video_siglip, gazing_info=gazing_info)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, List
from einops import rearrange

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.semantic_filter import SemanticFilter


class SemanticAutoGazeWrapper(torch.nn.Module):
    """
    Combines AutoGaze with semantic filtering for text-conditioned patch selection.

    Modes:
      - "intersect": AutoGaze selects patches, then semantic head prunes them.
                     Final budget = gazing_ratio * semantic_keep_ratio.
      - "semantic_only": Bypass AutoGaze, use semantic scores alone.
      - "gaze_only": Standard AutoGaze, no semantic filtering.
    """

    def __init__(
        self,
        autogaze_model_name: str = "bfshi/AutoGaze",
        head_ckpt: str = "results/distill_bighead/best_bighead_student.pt",
        head_type: str = "bighead",
        device: str = "cuda:0",
        use_flash_attn: bool = False,
        grid_size: int = 14,
        num_frames: int = 16,
        hidden_source: str = "post_decoder",
        **head_kwargs,
    ):
        super().__init__()
        if hidden_source not in ("post_decoder", "pre_decoder"):
            raise ValueError(f"hidden_source must be 'post_decoder' or 'pre_decoder', got {hidden_source!r}")
        self.device = torch.device(device)
        self.grid_size = grid_size
        self.num_frames = num_frames
        self.patches_per_frame = grid_size * grid_size  # 196
        self.hidden_source = hidden_source

        # Load AutoGaze
        self.autogaze = AutoGaze.from_pretrained(
            autogaze_model_name, use_flash_attn=use_flash_attn
        ).to(self.device).eval()

        # Load semantic filtering head
        self.semantic_filter = SemanticFilter(
            head_ckpt=head_ckpt,
            head_type=head_type,
            device=str(self.device),
            grid_size=grid_size,
            num_frames=num_frames,
            **head_kwargs,
        )

    @torch.no_grad()
    def extract_hidden_states(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract AutoGaze decoder hidden states for semantic scoring.

        Args:
            video: (B, T, C, H, W) preprocessed for AutoGaze

        Returns:
            hidden_states: (B, T*196, hidden_dim) decoder output features
        """
        gaze_model = self.autogaze.gazing_model
        B, T = video.shape[:2]

        # Resize to gaze model input size
        video_resized = rearrange(video, 'b t c h w -> (b t) c h w')
        video_resized = F.interpolate(
            video_resized,
            size=(gaze_model.input_img_size, gaze_model.input_img_size),
            mode="bicubic", align_corners=False,
        )
        video_resized = rearrange(video_resized, '(b t) c h w -> b t c h w', b=B)

        # Forward through vision model + connector
        vision_features, _ = gaze_model.vision_model(video_resized)
        vision_features = vision_features.transpose(1, 2)
        vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
        vision_features = gaze_model.connector(vision_features)

        B2, T2, N, C = vision_features.shape
        if self.hidden_source == "pre_decoder":
            return vision_features.reshape(B2, T2 * N, C)

        inputs_embeds = vision_features.reshape(B2, T2 * N, C)
        attention_mask = torch.ones(B2, T2 * N, device=self.device, dtype=torch.long)
        decoder_outputs = gaze_model.gaze_decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=attention_mask.cumsum(dim=-1) - 1,
        )

        return decoder_outputs.last_hidden_state  # (B, T*196, hidden_dim)

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        text_embedding: torch.Tensor,
        mode: str = "intersect",
        gazing_ratio: float = 0.75,
        task_loss_requirement: Optional[float] = 0.7,
        semantic_keep_ratio: float = 0.5,
        semantic_threshold: Optional[float] = None,
        target_scales: Optional[list] = None,
        target_patch_size: Optional[int] = None,
    ) -> dict:
        """
        Run the full SemanticAutoGaze pipeline.

        Args:
            video: (B, T, C, H, W) preprocessed for AutoGaze
            text_embedding: (B, 512) CLIP text embedding
            mode: "intersect" | "semantic_only" | "gaze_only"
            gazing_ratio: AutoGaze budget (fraction of patches)
            task_loss_requirement: AutoGaze task loss threshold
            semantic_keep_ratio: fraction of (gazed) patches to keep by semantic score
            semantic_threshold: absolute threshold (overrides semantic_keep_ratio)
            target_scales: vision encoder scales (for resolution adaptation)
            target_patch_size: vision encoder patch size

        Returns:
            gazing_info dict compatible with SigLIP's mask_with_gazing()
        """
        B = video.shape[0]

        if mode == "gaze_only":
            # Standard AutoGaze, no semantic filtering
            gaze_outputs = self.autogaze(
                {"video": video},
                gazing_ratio=gazing_ratio,
                task_loss_requirement=task_loss_requirement,
                generate_only=True,
                target_scales=target_scales,
                target_patch_size=target_patch_size,
            )
            return gaze_outputs

        # Extract hidden states for semantic scoring
        hidden_states = self.extract_hidden_states(video)

        # Compute semantic scores
        semantic_scores = self.semantic_filter.get_scores(
            hidden_states, text_embedding
        )

        if mode == "semantic_only":
            # Pure semantic filtering, bypass AutoGaze gaze
            gazing_info = self.semantic_filter.scores_to_gazing_info(
                semantic_scores,
                keep_ratio=semantic_keep_ratio,
                threshold=semantic_threshold,
                num_frames=self.num_frames,
            )
            return gazing_info

        elif mode == "intersect":
            # Run AutoGaze first, then filter by semantic score
            gaze_outputs = self.autogaze(
                {"video": video},
                gazing_ratio=gazing_ratio,
                task_loss_requirement=task_loss_requirement,
                generate_only=True,
                target_scales=target_scales,
                target_patch_size=target_patch_size,
            )

            # Intersect gaze selection with semantic scores
            filtered_info = self.semantic_filter.intersect_with_gaze(
                semantic_scores,
                gaze_outputs,
                semantic_keep_ratio=semantic_keep_ratio,
            )

            # Preserve AutoGaze metadata
            filtered_info["scales"] = gaze_outputs.get("scales")
            filtered_info["frame_sampling_rate"] = gaze_outputs.get("frame_sampling_rate")
            filtered_info["num_vision_tokens_each_frame"] = gaze_outputs.get("num_vision_tokens_each_frame")

            return filtered_info

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'intersect', 'semantic_only', or 'gaze_only'.")

    def get_stats(
        self,
        gaze_outputs: dict,
        filtered_outputs: dict,
    ) -> dict:
        """Compare token counts between original gaze and filtered outputs."""
        orig_valid = (~gaze_outputs["if_padded_gazing"]).sum().item()
        filt_valid = (~filtered_outputs["if_padded_gazing"]).sum().item()

        return {
            "original_tokens": orig_valid,
            "filtered_tokens": filt_valid,
            "tokens_saved": orig_valid - filt_valid,
            "reduction_pct": (1 - filt_valid / max(orig_valid, 1)) * 100,
        }
