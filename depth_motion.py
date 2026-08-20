"""
Depth-based image-to-motion generator.

Uses Depth Anything V2 Small for monocular depth estimation (when available),
then creates parallax motion by displacing pixels proportional to their depth —
closer objects move more, distant ones less — producing a 2-second 3D parallax
video.

Falls back to a gradient-based pseudo-depth estimator (OpenCV only) when the
ML model isn't installed, so the tool works out of the box.

Motion modes:
  dolly    — camera push-in / pull-out (zoom parallax)
  pan      — horizontal camera slide (left-right parallax)
  tilt     — vertical camera slide (up-down parallax)
  orbit    — circular camera orbit (combines pan + tilt)
  breathe  — gentle depth-based pulsing (3D breathing effect)

Usage:
  python depth_motion.py photo.jpg -o parallax.mp4 --mode dolly
  python depth_motion.py photo.jpg --mode orbit --amplitude 15 --duration 3
  python depth_motion.py photo.jpg --depth-method gradient  # no ML needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Depth estimation — ML model (Depth Anything V2 Small)
# ---------------------------------------------------------------------------

_depth_pipe = None


def _get_depth_pipeline():
    global _depth_pipe
    if _depth_pipe is not None:
        return _depth_pipe

    from transformers import pipeline

    print("Loading Depth Anything V2 Small model...")
    _depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    print("Model loaded.")
    return _depth_pipe


def estimate_depth_ml(image_bgr: np.ndarray) -> np.ndarray:
    """Run ML depth estimation. Returns float32 depth map [0, 1], higher = closer."""
    from PIL import Image

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    pipe = _get_depth_pipeline()
    result = pipe(pil_img)
    depth = np.array(result["depth"], dtype=np.float32)

    if depth.shape[:2] != image_bgr.shape[:2]:
        depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]))

    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth


# ---------------------------------------------------------------------------
# Depth estimation — gradient-based fallback (no ML, pure OpenCV)
# ---------------------------------------------------------------------------

def estimate_depth_gradient(image_bgr: np.ndarray) -> np.ndarray:
    """Estimate pseudo-depth using image gradients + vertical position heuristic.

    Combines:
    - Vertical position bias (bottom = closer, common in photos)
    - Edge/texture density (textured areas appear closer)
    - Brightness (brighter often = sky/far)

    Not as accurate as ML, but produces usable parallax for most photos.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. Vertical position — bottom of image tends to be closer
    y_coords = np.linspace(0, 1, h).reshape(-1, 1).astype(np.float32)
    vert_depth = np.tile(y_coords, (1, w))

    # 2. Texture/edge density — more detail often = closer
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    texture = cv2.GaussianBlur(laplacian, (31, 31), 0)
    t_max = texture.max()
    if t_max > 0:
        texture /= t_max

    # 3. Inverse brightness — darker often = foreground in many compositions
    brightness = gray / 255.0
    inv_bright = 1.0 - cv2.GaussianBlur(brightness, (31, 31), 0)

    # Combine with weights
    depth = 0.5 * vert_depth + 0.3 * texture + 0.2 * inv_bright

    # Normalize
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)

    # Smooth for natural transitions
    depth = cv2.GaussianBlur(depth, (21, 21), 0)

    return depth


def estimate_depth(image_bgr: np.ndarray, method: str = "auto") -> np.ndarray:
    """Estimate depth using the specified method.

    method: "auto" tries ML first and falls back to gradient,
            "ml" forces ML (errors if unavailable),
            "gradient" uses the pure-OpenCV fallback.
    """
    if method == "gradient":
        return estimate_depth_gradient(image_bgr)

    if method == "ml":
        return estimate_depth_ml(image_bgr)

    # Auto: try ML, fall back to gradient
    try:
        return estimate_depth_ml(image_bgr)
    except Exception as e:
        print(f"ML depth estimation unavailable ({e.__class__.__name__}), using gradient fallback.")
        return estimate_depth_gradient(image_bgr)


# ---------------------------------------------------------------------------
# Parallax displacement modes
# ---------------------------------------------------------------------------

def _dolly_displacement(
    depth: np.ndarray, h: int, w: int,
    frame: int, total_frames: int, amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera push-in/pull-out — closer objects shift outward from center."""
    t = frame / total_frames * 2 * np.pi
    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    strength = amplitude * np.sin(t) * depth
    dx = strength * (x_coords - cx) / (max(w, h) * 0.5)
    dy = strength * (y_coords - cy) / (max(w, h) * 0.5)
    return dx.astype(np.float32), dy.astype(np.float32)


def _pan_displacement(
    depth: np.ndarray, h: int, w: int,
    frame: int, total_frames: int, amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal camera slide — depth-proportional horizontal shift."""
    t = frame / total_frames * 2 * np.pi
    dx = (amplitude * np.sin(t) * depth).astype(np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    return dx, dy


def _tilt_displacement(
    depth: np.ndarray, h: int, w: int,
    frame: int, total_frames: int, amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical camera slide — depth-proportional vertical shift."""
    t = frame / total_frames * 2 * np.pi
    dx = np.zeros((h, w), dtype=np.float32)
    dy = (amplitude * np.sin(t) * depth).astype(np.float32)
    return dx, dy


def _orbit_displacement(
    depth: np.ndarray, h: int, w: int,
    frame: int, total_frames: int, amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Circular camera orbit — combines horizontal and vertical parallax."""
    t = frame / total_frames * 2 * np.pi
    dx = (amplitude * np.sin(t) * depth).astype(np.float32)
    dy = (amplitude * 0.5 * np.cos(t) * depth).astype(np.float32)
    return dx, dy


def _breathe_displacement(
    depth: np.ndarray, h: int, w: int,
    frame: int, total_frames: int, amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-based breathing — foreground expands/contracts from center."""
    t = frame / total_frames * 2 * np.pi
    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    strength = amplitude * 0.5 * np.sin(t) * (depth ** 1.5)
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) + 1e-6
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    norm_dist = dist / max_dist

    dx = (strength * norm_dist * (x_coords - cx) / dist).astype(np.float32)
    dy = (strength * norm_dist * (y_coords - cy) / dist).astype(np.float32)
    return dx, dy


MODES = {
    "dolly": _dolly_displacement,
    "pan": _pan_displacement,
    "tilt": _tilt_displacement,
    "orbit": _orbit_displacement,
    "breathe": _breathe_displacement,
}


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def render_parallax_frames(
    img: np.ndarray,
    depth: np.ndarray,
    mode: str = "dolly",
    duration: float = 2.0,
    fps: int = 24,
    amplitude: float = 12.0,
) -> list[np.ndarray]:
    """Render parallax motion frames using depth-based displacement."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODES.keys())}")

    h, w = img.shape[:2]
    total_frames = int(duration * fps)
    base_y, base_x = np.mgrid[0:h, 0:w].astype(np.float32)
    displace_fn = MODES[mode]

    frames = []
    for i in range(total_frames):
        dx, dy = displace_fn(depth, h, w, i, total_frames, amplitude)
        map_x = (base_x + dx).astype(np.float32)
        map_y = (base_y + dy).astype(np.float32)
        warped = cv2.remap(
            img, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        frames.append(warped)

        pct = (i + 1) / total_frames * 100
        print(f"\rRendering: {pct:.0f}%", end="", flush=True)

    print()
    return frames


def write_mp4(frames: list[np.ndarray], output_path: str, fps: int) -> str:
    """Write frames to MP4. Tries imageio+pyav first, falls back to OpenCV."""
    try:
        import imageio.v3 as iio

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        iio.imwrite(
            output_path,
            np.stack(rgb_frames),
            fps=fps,
            codec="libx264",
            plugin="pyav",
        )
        return output_path
    except (ImportError, Exception):
        pass

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return output_path


def generate_depth_motion(
    image_path: str,
    output_path: str = "parallax.mp4",
    mode: str = "dolly",
    duration: float = 2.0,
    fps: int = 24,
    amplitude: float = 12.0,
    depth_method: str = "auto",
    save_depth: bool = False,
) -> str:
    """Full pipeline: load image → estimate depth → render → export MP4."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        print(f"Resized to {img.shape[1]}x{img.shape[0]}")

    print(f"Estimating depth (method={depth_method})...")
    depth = estimate_depth(img, method=depth_method)

    if save_depth:
        depth_vis = (depth * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_path = str(Path(output_path).with_suffix("")) + "_depth.png"
        cv2.imwrite(depth_path, depth_vis)
        print(f"Depth map saved to {depth_path}")

    print(f"Rendering {duration}s @ {fps}fps, mode={mode}, amplitude={amplitude}...")
    frames = render_parallax_frames(img, depth, mode, duration, fps, amplitude)

    print("Writing MP4...")
    out = write_mp4(frames, output_path, fps)
    print(f"Done → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D parallax motion from a static image using depth estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", help="Input image (jpg/png)")
    parser.add_argument("-o", "--output", default="parallax.mp4", help="Output MP4 path (default: parallax.mp4)")
    parser.add_argument(
        "--mode", choices=list(MODES.keys()), default="dolly",
        help="Parallax mode (default: dolly)",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="Video duration in seconds (default: 2.0)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--amplitude", type=float, default=12.0, help="Motion strength (default: 12.0)")
    parser.add_argument(
        "--depth-method", choices=["auto", "ml", "gradient"], default="auto",
        help="Depth estimation method: auto (try ML, fallback to gradient), ml (Depth Anything V2), gradient (OpenCV only)",
    )
    parser.add_argument("--save-depth", action="store_true", help="Save depth map visualization as PNG")

    args = parser.parse_args()

    generate_depth_motion(
        image_path=args.image,
        output_path=args.output,
        mode=args.mode,
        duration=args.duration,
        fps=args.fps,
        amplitude=args.amplitude,
        depth_method=args.depth_method,
        save_depth=args.save_depth,
    )


if __name__ == "__main__":
    main()
