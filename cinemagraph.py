"""
Cinemagraph generator — animate a masked region of a static image.

Takes an input image and a grayscale mask (white = animate, black = freeze),
applies looping displacement/warp to the masked area, and writes a seamless
~1-second video.

Supported motion presets:
  ripple   — concentric sine-wave displacement (water, puddles)
  flow     — directional horizontal drift (rivers, streams)
  breathe  — gentle radial expansion/contraction (foliage, fabric)
  cloud    — slow lateral + vertical drift with turbulence (sky, smoke)

Usage:
  python cinemagraph.py image.jpg mask.png -o out.mp4 --motion ripple
  python cinemagraph.py image.jpg mask.png --motion cloud --duration 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Displacement-map generators (all return float32 dx, dy arrays per frame)
# ---------------------------------------------------------------------------

def _ripple_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 6.0, frequency: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Concentric ripple emanating from the mask center."""
    t = frame / total_frames * 2 * np.pi
    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    phase = dist / frequency * 2 * np.pi - t
    envelope = amplitude * np.sin(t)
    dx = envelope * np.cos(phase) * (x_coords - cx) / (dist + 1e-6)
    dy = envelope * np.cos(phase) * (y_coords - cy) / (dist + 1e-6)
    return dx, dy


def _flow_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 8.0, wave_len: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal drift with a vertical sine wobble (river/stream)."""
    t = frame / total_frames * 2 * np.pi
    y_coords = np.arange(h, dtype=np.float32)
    dx_col = amplitude * np.sin(y_coords / wave_len * 2 * np.pi + t)
    dx = np.tile(dx_col[:, None], (1, w))
    dy = np.full((h, w), amplitude * 0.3 * np.sin(t), dtype=np.float32)
    return dx, dy


def _breathe_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gentle radial expand/contract (foliage, fabric)."""
    t = frame / total_frames * 2 * np.pi
    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    scale = amplitude * np.sin(t)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) + 1e-6
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    factor = scale * (dist / max_dist)
    dx = factor * (x_coords - cx) / (dist + 1e-6)
    dy = factor * (y_coords - cy) / (dist + 1e-6)
    return dx, dy


def _cloud_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow lateral drift with layered sine turbulence (clouds, smoke)."""
    t = frame / total_frames * 2 * np.pi
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (
        amplitude * np.sin(t)
        + amplitude * 0.3 * np.sin(y_coords / 40 + t * 2)
    ).astype(np.float32)
    dy = (
        amplitude * 0.4 * np.sin(x_coords / 50 + t * 1.5)
        + amplitude * 0.2 * np.cos(y_coords / 35 + t)
    ).astype(np.float32)
    return dx, dy


MOTIONS = {
    "ripple": _ripple_displacement,
    "flow": _flow_displacement,
    "breathe": _breathe_displacement,
    "cloud": _cloud_displacement,
}


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def generate_cinemagraph(
    image_path: str,
    mask_path: str,
    output_path: str = "output.mp4",
    motion: str = "ripple",
    duration: float = 1.0,
    fps: int = 24,
    amplitude: float | None = None,
    feather: int = 5,
) -> str:
    """Generate a looping cinemagraph video.

    Returns the path of the written file.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    if mask_raw.shape[:2] != img.shape[:2]:
        mask_raw = cv2.resize(mask_raw, (img.shape[1], img.shape[0]))

    mask = mask_raw.astype(np.float32) / 255.0
    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask3 = np.stack([mask] * 3, axis=-1)

    if motion not in MOTIONS:
        raise ValueError(f"Unknown motion '{motion}'. Choose from: {list(MOTIONS.keys())}")
    displace_fn = MOTIONS[motion]

    h, w = img.shape[:2]
    total_frames = int(duration * fps)
    base_y, base_x = np.mgrid[0:h, 0:w].astype(np.float32)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    extra = {}
    if amplitude is not None:
        extra["amplitude"] = amplitude

    for i in range(total_frames):
        dx, dy = displace_fn(h, w, i, total_frames, **extra)
        map_x = (base_x + dx).astype(np.float32)
        map_y = (base_y + dy).astype(np.float32)
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        blended = (warped * mask3 + img * (1.0 - mask3)).astype(np.uint8)
        out.write(blended)

    out.release()
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a cinemagraph from a static image + mask.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", help="Input image (jpg/png)")
    parser.add_argument("mask", help="Grayscale mask — white regions will be animated")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path (default: output.mp4)")
    parser.add_argument(
        "--motion", choices=list(MOTIONS.keys()), default="ripple",
        help="Motion preset (default: ripple)",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="Loop duration in seconds (default: 1.0)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--amplitude", type=float, default=None, help="Override motion amplitude")
    parser.add_argument("--feather", type=int, default=5, help="Mask edge feather radius (default: 5)")

    args = parser.parse_args()

    path = generate_cinemagraph(
        image_path=args.image,
        mask_path=args.mask,
        output_path=args.output,
        motion=args.motion,
        duration=args.duration,
        fps=args.fps,
        amplitude=args.amplitude,
        feather=args.feather,
    )
    print(f"Cinemagraph saved to {path}")
    print(f"  {args.duration}s @ {args.fps}fps = {int(args.duration * args.fps)} frames, motion={args.motion}")


if __name__ == "__main__":
    main()
