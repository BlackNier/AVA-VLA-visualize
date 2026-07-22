"""CALVIN-specific renderer for saved AVA-VLA attention snapshots.

Renders each policy-query snapshot and optionally creates per-subtask videos
and one concatenated video for every sequence.  A CALVIN rollout can stop
before subtask 5 after a failure, so concatenation uses all subtask directories
that are actually present, in numeric order.
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from prismatic.attention_viz import (
    DEFAULT_OVERLAY_ALPHA,
    LAYER_GROUPS,
    make_attention_video,
    visualize_snapshot,
)


_STEP_DIR_RE = re.compile(r"step_(\d+)$")
_SUBTASK_RE = re.compile(r"subtask_(\d+)$")


def _matrix_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.rglob("step_*.pt"))


def _numeric_key(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.search(path.name)
    if match is None:
        return 10**9
    return int(match.group(1))


def _sequence_subtask(matrix_path: Path, input_dir: Path) -> Tuple[str, str]:
    relative_dir = matrix_path.parent.relative_to(input_dir)
    if len(relative_dir.parts) < 2:
        raise ValueError(
            f"Expected CALVIN layout sequence_xxxx/subtask_xx/step_xxxx.pt, got {matrix_path}"
        )
    return relative_dir.parts[0], relative_dir.parts[1]


def _render_snapshots(
    matrix_paths: Sequence[Path],
    input_dir: Path,
    output_dir: Path,
    layer_group: str,
    component: str,
    overlay_alpha: float,
) -> Dict[Tuple[str, str, str, str], List[Path]]:
    """Render snapshots and return frame paths grouped by sequence/subtask."""

    frames: Dict[Tuple[str, str, str, str], List[Path]] = defaultdict(list)
    components = ("raw", "final") if component == "both" else (component,)
    for matrix_path in matrix_paths:
        sequence_name, subtask_name = _sequence_subtask(matrix_path, input_dir)
        relative_dir = matrix_path.parent.relative_to(input_dir)
        snapshot_output_dir = output_dir / relative_dir / matrix_path.stem
        visualize_snapshot(
            matrix_path,
            snapshot_output_dir,
            layer_group,
            component,
            overlay_alpha=overlay_alpha,
        )
        for name in components:
            for view in ("full", "primary", "wrist"):
                frame_path = snapshot_output_dir / f"{name}_{layer_group}_{view}.png"
                if frame_path.exists():
                    frames[(sequence_name, subtask_name, name, view)].append(frame_path)

    for key in frames:
        frames[key].sort(key=lambda path: _numeric_key(path.parent, _STEP_DIR_RE))
    return frames


def _write_videos(
    frames: Dict[Tuple[str, str, str, str], List[Path]],
    output_dir: Path,
    layer_group: str,
    video_view: str,
    video_fps: int,
    concat_sequences: bool,
) -> None:
    components = sorted({key[2] for key in frames})
    views = ("full", "primary", "wrist") if video_view == "all" else (video_view,)

    for (sequence_name, subtask_name, name, view), frame_paths in sorted(frames.items()):
        if name not in components or view not in views:
            continue
        make_attention_video(
            frame_paths,
            output_dir / sequence_name / subtask_name / f"{name}_{layer_group}_{view}.mp4",
            video_fps,
        )

    if not concat_sequences:
        return

    grouped: Dict[Tuple[str, str, str], List[Tuple[str, List[Path]]]] = defaultdict(list)
    for (sequence_name, subtask_name, name, view), frame_paths in frames.items():
        if view in views:
            grouped[(sequence_name, name, view)].append((subtask_name, frame_paths))

    for (sequence_name, name, view), subtask_frames in sorted(grouped.items()):
        subtask_frames.sort(key=lambda item: _numeric_key(Path(item[0]), _SUBTASK_RE))
        if len(subtask_frames) != 5:
            warnings.warn(
                f"{sequence_name} contains {len(subtask_frames)} rendered subtasks; "
                "the sequence video will concatenate only those present."
            )
        all_frames = [frame for _, frames_for_subtask in subtask_frames for frame in frames_for_subtask]
        make_attention_video(
            all_frames,
            output_dir / sequence_name / f"{name}_{layer_group}_{view}_sequence.mp4",
            video_fps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and concatenate CALVIN attention snapshots.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--layer_group", choices=sorted(LAYER_GROUPS), default="L31")
    parser.add_argument("--component", choices=("raw", "final", "both"), default="both")
    parser.add_argument("--overlay_alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--concat_sequences", action="store_true", help="Concatenate subtask videos per sequence.")
    parser.add_argument("--video_view", choices=("full", "primary", "wrist", "all"), default="primary")
    parser.add_argument("--video_fps", type=int, default=5)
    args = parser.parse_args()

    if args.concat_sequences and not args.make_video:
        parser.error("--concat_sequences requires --make_video")
    if args.video_fps <= 0:
        parser.error("--video_fps must be positive")

    matrix_paths = list(_matrix_files(args.input_dir))
    if not matrix_paths:
        raise FileNotFoundError(f"No step_*.pt files found below {args.input_dir}")

    frames = _render_snapshots(
        matrix_paths,
        args.input_dir,
        args.output_dir,
        args.layer_group,
        args.component,
        args.overlay_alpha,
    )
    if args.make_video:
        _write_videos(
            frames,
            args.output_dir,
            args.layer_group,
            args.video_view,
            args.video_fps,
            args.concat_sequences,
        )


if __name__ == "__main__":
    main()
