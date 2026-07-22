"""Capture and visualize AVA-VLA attention matrices.

The AVA-VLA transformers fork applies ``extra_attn_weights`` inside the
eager Llama attention implementation.  This module wraps that implementation
at runtime and runs the same attention operation once without the AVA weights
and once with them, preserving the exact raw/final matrices without changing
model parameters or the external transformers fork.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image


LAYER_GROUPS = {
    "L30": (30,),
    "L31": (31,),
    "L28-L31": tuple(range(28, 32)),
    "L0-L31": tuple(range(0, 32)),
    "L16-L23": tuple(range(16, 24)),
}
DEFAULT_OVERLAY_ALPHA = 0.30


def _to_cpu_half(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().to(device="cpu", dtype=torch.float16)


class AttentionCapture(AbstractContextManager):
    """Temporarily capture raw and AVA-modulated Llama attention matrices."""

    def __init__(self, layers: Optional[Sequence[int]] = None) -> None:
        self.layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.selected_layers = None if layers is None else frozenset(int(layer) for layer in layers)
        self.layout: Dict[str, Any] = {}
        self._original_forward = None
        self._attention_class = None
        self._busy = False

    def __enter__(self) -> "AttentionCapture":
        from transformers.models.llama.modeling_llama import LlamaAttention

        self._attention_class = LlamaAttention
        self._original_forward = LlamaAttention.forward
        capture = self
        original_forward = self._original_forward

        def wrapped_forward(module, *args, **kwargs):
            # The recursive raw call must not trigger a second capture.
            layer_idx = getattr(module, "layer_idx", None)
            if (
                capture._busy
                or not kwargs.get("output_attentions", False)
                or (capture.selected_layers is not None and layer_idx not in capture.selected_layers)
            ):
                return original_forward(module, *args, **kwargs)

            extra_attn_weights = kwargs.get("extra_attn_weights")
            if extra_attn_weights is None:
                final_output = original_forward(module, *args, **kwargs)
                capture._store(module.layer_idx, final_output[1], final_output[1])
                return final_output

            raw_kwargs = dict(kwargs)
            raw_kwargs["extra_attn_weights"] = None
            capture._busy = True
            try:
                raw_output = original_forward(module, *args, **raw_kwargs)
            finally:
                capture._busy = False

            final_output = original_forward(module, *args, **kwargs)
            capture._store(module.layer_idx, raw_output[1], final_output[1])
            return final_output

        LlamaAttention.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._attention_class is not None and self._original_forward is not None:
            self._attention_class.forward = self._original_forward
        self._attention_class = None
        self._original_forward = None

    def _store(
        self,
        layer_idx: Optional[int],
        raw_attention: Optional[torch.Tensor],
        final_attention: Optional[torch.Tensor],
    ) -> None:
        if layer_idx is None:
            return
        if raw_attention is None or final_attention is None:
            raise RuntimeError(
                "Attention capture received no attention matrix. "
                "Make sure eager attention and output_attentions are enabled."
            )
        self.layers[int(layer_idx)] = {
            "raw": _to_cpu_half(raw_attention),
            "final": _to_cpu_half(final_attention),
        }

    def update_layout(self, **layout: Any) -> None:
        self.layout.update(layout)

    def snapshot(self) -> Dict[str, Any]:
        if not self.layers:
            raise RuntimeError("No attention matrices were captured.")
        captured_layers = set(self.layers)
        if self.selected_layers is not None and captured_layers != set(self.selected_layers):
            raise RuntimeError(
                f"Expected layers {sorted(self.selected_layers)}, got {sorted(captured_layers)}"
            )
        return {
            "raw": {layer: values["raw"] for layer, values in sorted(self.layers.items())},
            "final": {layer: values["final"] for layer, values in sorted(self.layers.items())},
            "layout": dict(self.layout),
            "captured_layers": sorted(captured_layers),
        }


def save_attention_snapshot(
    snapshot: Dict[str, Any],
    output_dir: Union[str, Path],
    step_name: str,
    primary_image: Optional[Any] = None,
    wrist_image: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save all layers plus model-view images for one LIBERO query step."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw": snapshot["raw"],
        "final": snapshot["final"],
        "layout": snapshot.get("layout", {}),
        "captured_layers": snapshot.get("captured_layers", sorted(snapshot["raw"])),
        "metadata": metadata or {},
    }
    matrix_path = output_dir / f"{step_name}.pt"
    torch.save(payload, matrix_path)

    for suffix, image in (("primary", primary_image), ("wrist", wrist_image)):
        if image is None:
            continue
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8))
        image.convert("RGB").save(output_dir / f"{step_name}_{suffix}.png")

    return matrix_path


def _mean_layer_matrix(payload: Dict[str, Any], component: str, layers: Sequence[int]) -> torch.Tensor:
    matrices = [payload[component][layer] for layer in layers]
    # [layers, batch, heads, query, key] -> [query, key]
    return torch.stack(matrices, dim=0).float().mean(dim=0).mean(dim=0).mean(dim=0)


def _overlay_attention(
    image_path: Path,
    scores: torch.Tensor,
    output_path: Path,
    title: str,
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> None:
    del title  # The output filename identifies the layer/component/view.
    _blend_attention_overlay(image_path, scores, overlay_alpha).save(output_path)


def _blend_attention_overlay(
    image_path: Path,
    scores: torch.Tensor,
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> Image.Image:
    """Return an RGB image with patch attention alpha-blended onto the input image."""

    if not 0.0 < overlay_alpha < 1.0:
        raise ValueError("overlay_alpha must be between 0 and 1.")
    image = Image.open(image_path).convert("RGB")
    height = int(scores.shape[0] ** 0.5)
    width = scores.shape[0] // height
    if height * width != scores.numel():
        raise ValueError(f"Visual token count {scores.numel()} is not a rectangular patch grid.")

    heatmap = scores.reshape(height, width).detach().cpu().numpy().copy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8

    heatmap_rgb = (plt.get_cmap("jet")(heatmap)[..., :3] * 255).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    heatmap_image = Image.fromarray(heatmap_rgb).resize(image.size, resampling)
    return Image.blend(image, heatmap_image, alpha=overlay_alpha)


def _full_matrix_image(matrix: torch.Tensor, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix.numpy(), cmap="magma", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("key token")
    plt.ylabel("query token")
    plt.title(title)
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close()


def visualize_snapshot(
    matrix_path: Union[str, Path],
    output_dir: Union[str, Path],
    layer_group: str,
    component: str = "both",
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> None:
    """Render full matrices and action-to-vision overlays for one snapshot."""

    if layer_group not in LAYER_GROUPS:
        raise ValueError(f"Unknown layer group {layer_group}; choose from {sorted(LAYER_GROUPS)}")
    if component not in {"raw", "final", "both"}:
        raise ValueError("component must be raw, final, or both")

    matrix_path = Path(matrix_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(matrix_path, map_location="cpu")
    layout = payload["layout"]
    layers = LAYER_GROUPS[layer_group]
    available_layers = set(payload["raw"].keys())
    if not set(layers).issubset(available_layers):
        raise ValueError(f"Snapshot contains layers {sorted(available_layers)}, but {layer_group} was requested.")

    query_start = int(layout["action_token_start"])
    query_count = int(layout["action_token_count"])
    visual_start = int(layout["visual_token_start"])
    visual_count = int(layout["visual_token_count"])
    num_images = int(layout["num_images"])
    patches_per_image = visual_count // num_images
    image_names = ["primary"] + (["wrist"] if num_images > 1 else [])

    components = ("raw", "final") if component == "both" else (component,)
    for name in components:
        matrix = _mean_layer_matrix(payload, name, layers)
        _full_matrix_image(matrix, output_dir / f"{name}_{layer_group}_full.png", f"{name} {layer_group}")

        action_to_visual = matrix[query_start : query_start + query_count, visual_start : visual_start + visual_count]
        action_to_visual = action_to_visual.mean(dim=0)
        for image_idx, image_name in enumerate(image_names):
            start = image_idx * patches_per_image
            end = start + patches_per_image
            image_path = matrix_path.parent / f"{matrix_path.stem}_{image_name}.png"
            if not image_path.exists():
                continue
            _overlay_attention(
                image_path,
                action_to_visual[start:end],
                output_dir / f"{name}_{layer_group}_{image_name}.png",
                f"{name} {layer_group} action-to-vision ({image_name})",
                overlay_alpha=overlay_alpha,
            )


def _iter_matrix_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.rglob("step_*.pt"))


def make_attention_video(frame_paths: Sequence[Path], output_path: Path, fps: int) -> None:
    """Encode rendered attention frames as an MP4 video."""

    if not frame_paths:
        raise ValueError("Cannot create an attention video without frames.")
    if fps <= 0:
        raise ValueError("Video FPS must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        first_height, first_width = None, None
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            if first_height is None:
                first_height, first_width = frame.shape[:2]
            elif frame.shape[:2] != (first_height, first_width):
                frame = np.asarray(Image.fromarray(frame).resize((first_width, first_height)))
            writer.append_data(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved AVA-VLA attention matrices.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--layer_group", choices=sorted(LAYER_GROUPS), default="L30")
    parser.add_argument("--component", choices=("raw", "final", "both"), default="both")
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=DEFAULT_OVERLAY_ALPHA,
        help="Attention overlay opacity for primary/wrist PNGs and videos (0 to 1).",
    )
    parser.add_argument(
        "--make_video",
        action="store_true",
        help="Also encode rendered step frames into MP4 videos.",
    )
    parser.add_argument(
        "--video_view",
        choices=("full", "primary", "wrist", "all"),
        default="primary",
        help="Frame type used for videos; 'all' writes every available view.",
    )
    parser.add_argument("--video_fps", type=int, default=5)
    args = parser.parse_args()

    files = list(_iter_matrix_files(args.input_dir))
    if not files:
        raise FileNotFoundError(f"No step_*.pt files found below {args.input_dir}")
    rendered_frames: Dict[Tuple[Path, str, str], List[Path]] = {}
    for matrix_path in files:
        relative_dir = matrix_path.parent.relative_to(args.input_dir)
        visualize_snapshot(
            matrix_path,
            args.output_dir / relative_dir / matrix_path.stem,
            args.layer_group,
            args.component,
            overlay_alpha=args.overlay_alpha,
        )

        if args.make_video:
            components = ("raw", "final") if args.component == "both" else (args.component,)
            views = ("full", "primary", "wrist") if args.video_view == "all" else (args.video_view,)
            for name in components:
                for view in views:
                    frame_path = (
                        args.output_dir
                        / relative_dir
                        / matrix_path.stem
                        / f"{name}_{args.layer_group}_{view}.png"
                    )
                    if frame_path.exists():
                        rendered_frames.setdefault((relative_dir, name, view), []).append(frame_path)

    if args.make_video:
        for (relative_dir, name, view), frame_paths in rendered_frames.items():
            make_attention_video(
                frame_paths,
                args.output_dir / relative_dir / f"{name}_{args.layer_group}_{view}.mp4",
                args.video_fps,
            )


if __name__ == "__main__":
    main()
