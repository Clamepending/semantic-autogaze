"""Export DINOv2-small + adapter to ONNX with INT8 dynamic quantization for Raspberry Pi."""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


class DINOv2PatchEncoder(nn.Module):
    """Wraps DINOv2 to output only L2-normalized patch features (no CLS)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        patch_features = outputs.last_hidden_state[:, 1:]
        return F.normalize(patch_features, dim=-1)


class TextAdapter(nn.Module):
    """The small MLP adapter: takes CLIP text features, outputs DINO-space query."""

    def __init__(self, clip_dim: int, dino_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            self.proj = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, dino_dim),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dino_dim),
            )
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(text_features), dim=-1)


def export_dino(output_dir: Path, dino_size: int = 448):
    print("Loading DINOv2-small...")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    encoder = DINOv2PatchEncoder(dino_model).eval()

    dummy_input = torch.randn(1, 3, dino_size, dino_size)
    onnx_path = output_dir / "dinov2_small_patches.onnx"

    print(f"Exporting DINOv2 to ONNX ({onnx_path})...")
    torch.onnx.export(
        encoder,
        dummy_input,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["patch_features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "patch_features": {0: "batch"}},
        opset_version=17,
    )

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"  ONNX model valid. Size: {onnx_path.stat().st_size / 1e6:.1f} MB")

    quant_path = output_dir / "dinov2_small_patches_int8.onnx"
    print(f"Quantizing to INT8 ({quant_path})...")
    quantize_dynamic(
        str(onnx_path),
        str(quant_path),
        weight_type=QuantType.QInt8,
    )
    print(f"  Quantized size: {quant_path.stat().st_size / 1e6:.1f} MB")
    return onnx_path, quant_path


def export_adapter(output_dir: Path, adapter_pt: Path):
    print(f"Loading adapter from {adapter_pt}...")
    ckpt = torch.load(adapter_pt, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    hidden_dim = config.get("hidden_dim", None)
    clip_dim = 512
    dino_dim = 384

    adapter = TextAdapter(clip_dim, dino_dim, hidden_dim=hidden_dim)
    state = ckpt["adapter"]
    adapter.proj.load_state_dict({k.replace("proj.", ""): v for k, v in state.items() if k.startswith("proj.")})
    adapter.logit_scale.data = state["logit_scale"]
    adapter.eval()

    logit_scale = float(adapter.logit_scale.clamp(1.0, 100.0).item())

    dummy_text = torch.randn(1, clip_dim)
    onnx_path = output_dir / "text_adapter.onnx"

    print(f"Exporting adapter to ONNX ({onnx_path})...")
    torch.onnx.export(
        adapter,
        dummy_text,
        str(onnx_path),
        input_names=["text_features"],
        output_names=["query"],
        dynamic_axes={"text_features": {0: "batch"}, "query": {0: "batch"}},
        opset_version=17,
    )
    print(f"  Adapter ONNX size: {onnx_path.stat().st_size / 1e3:.1f} KB")

    np.savez(
        output_dir / "adapter_meta.npz",
        logit_scale=np.array([logit_scale], dtype=np.float32),
        dino_size=np.array([config["dino_size"]], dtype=np.int32),
    )
    return onnx_path, logit_scale


def export_clip_text(output_dir: Path):
    """Pre-compute is better for RPi, but export CLIP text encoder for completeness."""
    print("Loading CLIP text encoder...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    print("Pre-computing common category embeddings...")
    categories = [
        "person", "car", "cat", "dog", "cup", "bottle", "chair", "table",
        "phone", "laptop", "book", "plant", "bicycle", "bus", "truck",
        "bird", "horse", "sheep", "cow", "bear", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "wine glass", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "couch", "bed", "toilet",
        "tv", "remote", "keyboard", "microwave", "oven", "toaster", "sink",
        "refrigerator", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
    texts = [f"a photo of a {c}" for c in categories]
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model.text_model(**inputs)
        features = clip_model.text_projection(outputs.pooler_output)
        features = F.normalize(features, dim=-1).numpy()

    np.savez(
        output_dir / "clip_text_embeddings.npz",
        categories=np.array(categories),
        features=features,
    )
    print(f"  Saved {len(categories)} pre-computed text embeddings")
    return categories


def validate(output_dir: Path, dino_size: int = 448):
    """Quick validation that ONNX outputs match PyTorch."""
    import onnxruntime as ort

    print("\nValidating ONNX vs PyTorch...")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    encoder = DINOv2PatchEncoder(dino_model).eval()

    x = torch.randn(1, 3, dino_size, dino_size)
    with torch.no_grad():
        pt_out = encoder(x).numpy()

    sess = ort.InferenceSession(str(output_dir / "dinov2_small_patches.onnx"))
    ort_out = sess.run(None, {"pixel_values": x.numpy()})[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"  FP32 ONNX max diff: {diff:.6f}")

    sess_q = ort.InferenceSession(str(output_dir / "dinov2_small_patches_int8.onnx"))
    ort_q_out = sess_q.run(None, {"pixel_values": x.numpy()})[0]

    diff_q = np.abs(pt_out - ort_q_out).max()
    print(f"  INT8 ONNX max diff: {diff_q:.6f}")
    print(f"  Cosine similarity (FP32 vs INT8): {np.mean(np.sum(ort_out * ort_q_out, axis=-1)):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Export DINOv2 + adapter to ONNX INT8")
    parser.add_argument("--adapter", default="models/clip_text_to_dino_adapter_coco_10k_448_mlp.pt")
    parser.add_argument("--output-dir", default="models/onnx_int8")
    parser.add_argument("--dino-size", type=int, default=448)
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_dino(output_dir, args.dino_size)
    export_adapter(output_dir, Path(args.adapter))
    export_clip_text(output_dir)

    if args.validate:
        validate(output_dir, args.dino_size)

    print(f"\n=== Export complete → {output_dir}/ ===")
    print("Files for Raspberry Pi deployment:")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name:40s} {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
