"""Compare raw and AVA-final attention matrices saved in attention snapshots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


LAYER_GROUPS = {
    "L30": (30,),
    "L31": (31,),
    "L28-L31": tuple(range(28, 32)),
    "L0-L31": tuple(range(0, 32)),
    "L16-L23": tuple(range(16, 24)),
}


def _load_payload(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        return torch.load(path, map_location="cpu")


def _find_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("step_*.pt"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _get_layers(payload: Dict[str, Any], layer_group: str) -> Sequence[int]:
    if layer_group not in LAYER_GROUPS:
        raise ValueError(f"Unknown layer group {layer_group}; choose from {sorted(LAYER_GROUPS)}")

    raw_layers = set(payload.get("raw", {}).keys())
    final_layers = set(payload.get("final", {}).keys())
    if raw_layers != final_layers:
        raise ValueError(
            f"raw/final layer mismatch: raw={sorted(raw_layers)}, final={sorted(final_layers)}"
        )

    layers = LAYER_GROUPS[layer_group]
    missing = set(layers) - raw_layers
    if missing:
        raise ValueError(
            f"Requested {layer_group}, but {sorted(missing)} are missing; "
            f"available layers are {sorted(raw_layers)}"
        )
    return layers


def _compare_layer(raw: torch.Tensor, final: torch.Tensor, tolerance: float) -> Dict[str, float]:
    if raw.shape != final.shape:
        raise ValueError(f"Shape mismatch: raw={tuple(raw.shape)}, final={tuple(final.shape)}")
    if raw.ndim != 4:
        raise ValueError(
            f"Expected [batch, heads, query, key] attention matrix, got {tuple(raw.shape)}"
        )

    raw = raw.float()
    final = final.float()
    diff = (final - raw).abs()
    eps = 1e-8

    raw_rows = raw.reshape(-1, raw.shape[-1])
    final_rows = final.reshape(-1, final.shape[-1])
    cosine = F.cosine_similarity(raw_rows, final_rows, dim=-1, eps=eps)

    # KL is reported row-wise; clamp only for numerical stability after loading
    # float16 snapshots.
    raw_prob = raw.clamp_min(eps)
    final_prob = final.clamp_min(eps)
    kl_raw_to_final = (raw_prob * (raw_prob.log() - final_prob.log())).sum(dim=-1)

    return {
        "mean_abs_diff": diff.mean().item(),
        "max_abs_diff": diff.max().item(),
        "relative_mean_abs_diff": (diff.mean() / raw.abs().mean().clamp_min(eps)).item(),
        "changed_fraction": (diff > tolerance).float().mean().item(),
        "row_l1_diff": diff.sum(dim=-1).mean().item(),
        "row_cosine_similarity": cosine.mean().item(),
        "kl_raw_to_final": kl_raw_to_final.mean().item(),
    }


def compare_file(path: Path, layer_group: str, tolerance: float) -> List[Dict[str, Any]]:
    payload = _load_payload(path)
    if "raw" not in payload or "final" not in payload:
        raise ValueError(f"{path} does not contain both 'raw' and 'final' dictionaries")

    rows: List[Dict[str, Any]] = []
    for layer in _get_layers(payload, layer_group):
        metrics = _compare_layer(payload["raw"][layer], payload["final"][layer], tolerance)
        rows.append({"file": str(path), "layer": layer, **metrics})
    return rows


def _print_rows(rows: Iterable[Dict[str, Any]]) -> None:
    fields = (
        "file",
        "layer",
        "mean_abs_diff",
        "max_abs_diff",
        "relative_mean_abs_diff",
        "changed_fraction",
        "row_l1_diff",
        "row_cosine_similarity",
        "kl_raw_to_final",
    )
    print("\t".join(fields))
    for row in rows:
        print(
            "\t".join(
                str(row[field]) if field in {"file", "layer"} else f"{row[field]:.8g}"
                for field in fields
            )
        )


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw and final attention matrices inside saved .pt snapshots."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="One .pt snapshot or a directory containing step_*.pt snapshots.",
    )
    parser.add_argument(
        "--layer_group",
        choices=sorted(LAYER_GROUPS),
        default="L0-L31",
        help="Layers to compare.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Absolute difference threshold for changed_fraction.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional path for a CSV copy of the per-file, per-layer metrics.",
    )
    args = parser.parse_args()

    files = _find_files(args.input)
    if not files:
        raise FileNotFoundError(f"No step_*.pt files found below {args.input}")

    rows: List[Dict[str, Any]] = []
    for path in files:
        try:
            rows.extend(compare_file(path, args.layer_group, args.tolerance))
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Failed to compare {path}: {exc}") from exc

    _print_rows(rows)
    if args.output_csv is not None:
        _write_csv(args.output_csv, rows)
        print(f"\nWrote CSV metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
