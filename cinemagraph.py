"""
Cinemagraph generator — animate a masked region of a static image.

Takes an input image and a grayscale mask (white = animate, black = freeze),
applies looping displacement/warp to the masked area, and writes a seamless
~1-second video (MP4 or GIF).

Supported motion presets:
  ripple   — concentric sine-wave displacement (water, puddles)
  flow     — directional horizontal drift (rivers, streams)
  breathe  — gentle radial expansion/contraction (foliage, fabric)
  cloud    — slow lateral + vertical drift with turbulence (sky, smoke)
  sway     — pendulum swing left/right (trees, grass, hanging objects)
  shimmer  — high-frequency micro-jitter (heat haze, glitter, light on water)
  zoom     — pulsing zoom in/out from center (heartbeat, pulsating light)
  spiral   — rotational twist around center (whirlpool, vortex)

Auto-mask modes (use instead of providing a mask file):
  sky      — detect sky regions via HSV color analysis
  water    — detect water/blue regions
  green    — detect foliage/vegetation

Usage:
  python cinemagraph.py image.jpg mask.png -o out.mp4 --motion ripple
  python cinemagraph.py image.jpg --auto-mask water --motion flow -o river.gif
  python cinemagraph.py image.jpg --auto-mask sky --motion cloud --duration 2.0
"""

from __future__ import annotations

import argparse
import struct
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


def _sway_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pendulum swing — top stays fixed, bottom sways (trees, grass)."""
    t = frame / total_frames * 2 * np.pi
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    y_factor = (y_coords / h) ** 2
    dx = (amplitude * np.sin(t) * y_factor).astype(np.float32)
    dy = (amplitude * 0.15 * np.sin(t * 2) * y_factor).astype(np.float32)
    return dx, dy


def _shimmer_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """High-frequency micro-jitter (heat haze, glitter on water)."""
    t = frame / total_frames * 2 * np.pi
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (
        amplitude * np.sin(x_coords / 8 + t * 3)
        * np.cos(y_coords / 10 + t * 2)
    ).astype(np.float32)
    dy = (
        amplitude * np.cos(x_coords / 10 + t * 2.5)
        * np.sin(y_coords / 8 + t * 3.5)
    ).astype(np.float32)
    return dx, dy


def _zoom_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pulsing zoom in/out from center (heartbeat, pulsating glow)."""
    t = frame / total_frames * 2 * np.pi
    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    scale = amplitude * np.sin(t) / max(h, w)
    dx = (scale * (x_coords - cx)).astype(np.float32)
    dy = (scale * (y_coords - cy)).astype(np.float32)
    return dx, dy


def _spiral_displacement(
    h: int, w: int, frame: int, total_frames: int,
    amplitude: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotational twist around center (whirlpool, vortex)."""
    t = frame / total_frames * 2 * np.pi
    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    rel_x = x_coords - cx
    rel_y = y_coords - cy
    dist = np.sqrt(rel_x ** 2 + rel_y ** 2) + 1e-6
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    angle = amplitude * np.sin(t) * (1 - dist / max_dist) / max_dist
    dx = (rel_x * np.cos(angle) - rel_y * np.sin(angle) - rel_x).astype(np.float32)
    dy = (rel_x * np.sin(angle) + rel_y * np.cos(angle) - rel_y).astype(np.float32)
    return dx, dy


MOTIONS = {
    "ripple": _ripple_displacement,
    "flow": _flow_displacement,
    "breathe": _breathe_displacement,
    "cloud": _cloud_displacement,
    "sway": _sway_displacement,
    "shimmer": _shimmer_displacement,
    "zoom": _zoom_displacement,
    "spiral": _spiral_displacement,
}


# ---------------------------------------------------------------------------
# Auto-mask generation
# ---------------------------------------------------------------------------

def auto_mask_sky(image: np.ndarray) -> np.ndarray:
    """Detect sky regions using HSV hue/saturation + upper-image bias."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blue_mask = cv2.inRange(hsv, np.array([90, 20, 120]), np.array([140, 255, 255]))
    bright_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
    sky = cv2.bitwise_or(blue_mask, bright_mask)
    rows = image.shape[0]
    position_weight = np.linspace(1.0, 0.0, rows).reshape(-1, 1).astype(np.float32)
    position_mask = (position_weight * 255).astype(np.uint8)
    position_mask = np.tile(position_mask, (1, image.shape[1]))
    sky = cv2.bitwise_and(sky, position_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, kernel, iterations=2)
    sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, kernel, iterations=1)
    return sky


def auto_mask_water(image: np.ndarray) -> np.ndarray:
    """Detect water regions using HSV blue/cyan + lower-image bias."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    water1 = cv2.inRange(hsv, np.array([85, 30, 40]), np.array([135, 255, 255]))
    water2 = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 40, 200]))
    rows = image.shape[0]
    position_weight = np.linspace(0.0, 1.0, rows).reshape(-1, 1).astype(np.float32)
    position_mask = (position_weight * 255).astype(np.uint8)
    position_mask = np.tile(position_mask, (1, image.shape[1]))
    water2 = cv2.bitwise_and(water2, position_mask)
    water = cv2.bitwise_or(water1, water2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel, iterations=2)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel, iterations=1)
    return water


def auto_mask_green(image: np.ndarray) -> np.ndarray:
    """Detect green foliage/vegetation regions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)
    return green


AUTO_MASKS = {
    "sky": auto_mask_sky,
    "water": auto_mask_water,
    "green": auto_mask_green,
}


# ---------------------------------------------------------------------------
# GIF writer (pure-Python, no PIL dependency)
# ---------------------------------------------------------------------------

def _write_gif(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    """Write an animated GIF with looping. Uses median-cut quantization."""
    delay = max(1, round(100 / fps))

    def _quantize(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
        pixels = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, palette = cv2.kmeans(
            pixels, 256, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
        )
        palette = np.clip(palette, 0, 255).astype(np.uint8)
        flat = rgb.reshape(-1, 3).astype(np.float32)
        dists = np.linalg.norm(flat[:, None, :] - palette[None, :, :], axis=2)
        indices = np.argmin(dists, axis=1).astype(np.uint8)
        return indices.reshape(rgb.shape[:2]), palette

    h, w = frames[0].shape[:2]
    with open(output_path, "wb") as f:
        f.write(b"GIF89a")
        f.write(struct.pack("<HH", w, h))
        f.write(struct.pack("BBB", 0xF7, 0, 0))
        first_idx, first_pal = _quantize(frames[0])
        for c in first_pal:
            f.write(bytes(c))
        for _ in range(256 - len(first_pal)):
            f.write(b"\x00\x00\x00")
        f.write(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")

        for frame_bgr in frames:
            idx, _ = _quantize(frame_bgr)
            f.write(b"\x21\xF9\x04\x00")
            f.write(struct.pack("<H", delay))
            f.write(b"\x00\x00")
            f.write(b"\x2C")
            f.write(struct.pack("<HHHH", 0, 0, w, h))
            f.write(b"\x00")
            min_code_size = 8
            f.write(bytes([min_code_size]))
            flat = idx.flatten()
            from io import BytesIO
            buf = BytesIO()
            _lzw_compress(flat, min_code_size, buf)
            data = buf.getvalue()
            i = 0
            while i < len(data):
                chunk = data[i:i + 255]
                f.write(bytes([len(chunk)]))
                f.write(chunk)
                i += 255
            f.write(b"\x00")

        f.write(b"\x3B")


def _lzw_compress(data: np.ndarray, min_code_size: int, out) -> None:
    """LZW compress index data for GIF."""
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    code_size = min_code_size + 1
    next_code = eoi_code + 1
    max_code = (1 << code_size) - 1

    table: dict[tuple, int] = {}
    for i in range(clear_code):
        table[(i,)] = i

    bit_buf = 0
    bit_count = 0
    result = bytearray()

    def emit(code: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            result.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8

    emit(clear_code)
    w_tuple = (int(data[0]),)

    for px in data[1:]:
        k = int(px)
        wk = w_tuple + (k,)
        if wk in table:
            w_tuple = wk
        else:
            emit(table[w_tuple])
            if next_code <= 4095:
                table[wk] = next_code
                next_code += 1
                if next_code > max_code + 1 and code_size < 12:
                    code_size += 1
                    max_code = (1 << code_size) - 1
            else:
                emit(clear_code)
                table.clear()
                for i in range(clear_code):
                    table[(i,)] = i
                next_code = eoi_code + 1
                code_size = min_code_size + 1
                max_code = (1 << code_size) - 1
            w_tuple = (k,)

    emit(table[w_tuple])
    emit(eoi_code)
    if bit_count > 0:
        result.append(bit_buf & 0xFF)
    out.write(bytes(result))


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def render_frames(
    img: np.ndarray,
    mask_raw: np.ndarray,
    motion: str = "ripple",
    duration: float = 1.0,
    fps: int = 24,
    amplitude: float | None = None,
    feather: int = 5,
) -> list[np.ndarray]:
    """Render cinemagraph frames from image + mask arrays. Returns list of BGR frames."""
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

    extra = {}
    if amplitude is not None:
        extra["amplitude"] = amplitude

    frames = []
    for i in range(total_frames):
        dx, dy = displace_fn(h, w, i, total_frames, **extra)
        map_x = (base_x + dx).astype(np.float32)
        map_y = (base_y + dy).astype(np.float32)
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        blended = (warped * mask3 + img * (1.0 - mask3)).astype(np.uint8)
        frames.append(blended)
    return frames


def generate_cinemagraph(
    image_path: str,
    mask_path: str | None = None,
    output_path: str = "output.mp4",
    motion: str = "ripple",
    duration: float = 1.0,
    fps: int = 24,
    amplitude: float | None = None,
    feather: int = 5,
    auto_mask: str | None = None,
) -> str:
    """Generate a looping cinemagraph video or GIF.

    Returns the path of the written file.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if auto_mask:
        if auto_mask not in AUTO_MASKS:
            raise ValueError(f"Unknown auto-mask '{auto_mask}'. Choose from: {list(AUTO_MASKS.keys())}")
        mask_raw = AUTO_MASKS[auto_mask](img)
    elif mask_path:
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    else:
        raise ValueError("Provide either a mask file or --auto-mask")

    frames = render_frames(img, mask_raw, motion, duration, fps, amplitude, feather)

    if output_path.lower().endswith(".gif"):
        _write_gif(frames, output_path, fps)
    else:
        h, w = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()

    return output_path


def generate_auto_mask(image_path: str, mode: str) -> np.ndarray:
    """Generate an auto-mask from an image file. Returns grayscale mask array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if mode not in AUTO_MASKS:
        raise ValueError(f"Unknown auto-mask '{mode}'. Choose from: {list(AUTO_MASKS.keys())}")
    return AUTO_MASKS[mode](img)


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
    parser.add_argument("mask", nargs="?", default=None, help="Grayscale mask (optional if --auto-mask used)")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output path — .mp4 or .gif (default: output.mp4)")
    parser.add_argument(
        "--motion", choices=list(MOTIONS.keys()), default="ripple",
        help="Motion preset (default: ripple)",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="Loop duration in seconds (default: 1.0)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--amplitude", type=float, default=None, help="Override motion amplitude")
    parser.add_argument("--feather", type=int, default=5, help="Mask edge feather radius (default: 5)")
    parser.add_argument(
        "--auto-mask", choices=list(AUTO_MASKS.keys()), default=None,
        help="Auto-detect mask region instead of providing a mask file",
    )

    args = parser.parse_args()

    if not args.mask and not args.auto_mask:
        parser.error("Provide a mask file or use --auto-mask {sky,water,green}")

    path = generate_cinemagraph(
        image_path=args.image,
        mask_path=args.mask,
        output_path=args.output,
        motion=args.motion,
        duration=args.duration,
        fps=args.fps,
        amplitude=args.amplitude,
        feather=args.feather,
        auto_mask=args.auto_mask,
    )
    ext = Path(path).suffix.upper().lstrip(".")
    print(f"Cinemagraph saved to {path} ({ext})")
    print(f"  {args.duration}s @ {args.fps}fps = {int(args.duration * args.fps)} frames, motion={args.motion}")


if __name__ == "__main__":
    main()
