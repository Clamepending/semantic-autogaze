#!/usr/bin/env python3
"""
End-to-end script: train semantic similarity head and generate visualizations.

Usage:
    python run.py                          # Use default example video
    python run.py --video_dir /path/to/vids  # Use custom video directory
    python run.py --task_text "a red car"   # Custom semantic query
"""

import os
import sys
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Semantic AutoGaze: Train and Visualize")
    parser.add_argument("--autogaze_model", type=str, default="nvidia/AutoGaze")
    parser.add_argument("--siglip_model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None,
                       help="Video for visualization (default: example_input.mp4)")
    parser.add_argument("--task_text", type=str, default="a traffic light changing colors",
                       help="Text description of the task for semantic filtering")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--gaze_threshold", type=float, default=0.7)
    parser.add_argument("--similarity_threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--wandb_project", type=str, default="claude/semantic-autogaze")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_viz", action="store_true")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_dir = os.path.join(project_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Default video path
    if args.video_path is None:
        args.video_path = os.path.join(project_dir, "assets", "example_input.mp4")

    # ==================== STEP 1: Train ====================
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 1: Training Semantic Similarity Head")
        print("=" * 60)

        from semantic_autogaze.train import train

        class TrainArgs:
            pass

        train_args = TrainArgs()
        train_args.autogaze_model = args.autogaze_model
        train_args.siglip_model = args.siglip_model
        train_args.video_dir = args.video_dir
        train_args.output_dir = args.output_dir
        train_args.batch_size = args.batch_size
        train_args.num_epochs = args.num_epochs
        train_args.lr = args.lr
        train_args.num_frames = args.num_frames
        train_args.device = args.device
        train_args.wandb_project = args.wandb_project
        train_args.seed = 42

        train(train_args)
    else:
        print("Skipping training (--skip_train)")

    # ==================== STEP 2: Visualize ====================
    if not args.skip_viz:
        print("\n" + "=" * 60)
        print("STEP 2: Generating Visualizations")
        print("=" * 60)

        from semantic_autogaze.visualize import visualize

        class VizArgs:
            pass

        viz_args = VizArgs()
        viz_args.video_path = args.video_path
        viz_args.task_text = args.task_text
        viz_args.output_dir = args.output_dir
        viz_args.siglip_model = args.siglip_model
        viz_args.gaze_threshold = args.gaze_threshold
        viz_args.similarity_threshold = args.similarity_threshold
        viz_args.num_frames = args.num_frames
        viz_args.device = args.device

        visualize(viz_args)
    else:
        print("Skipping visualization (--skip_viz)")

    print("\n" + "=" * 60)
    print("DONE! All outputs saved to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
