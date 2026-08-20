"""
Web UI for the cinemagraph generator.

Launch:
  pip install -r requirements.txt
  python app.py

Opens a Gradio interface where you can:
  - Upload an image
  - Draw a mask or use auto-detection (sky/water/green)
  - Pick a motion preset and adjust parameters
  - Preview and download the result as MP4 or GIF
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from cinemagraph import (
    AUTO_MASKS,
    MOTIONS,
    generate_auto_mask,
    render_frames,
    _write_gif,
)


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Return image with mask region tinted for preview."""
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * (1 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def preview_auto_mask(image: np.ndarray, auto_mask_mode: str) -> np.ndarray | None:
    if image is None:
        return None
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if auto_mask_mode not in AUTO_MASKS:
        return image
    mask = AUTO_MASKS[auto_mask_mode](bgr)
    overlay = _overlay_mask(bgr, mask)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def run_cinemagraph(
    image: np.ndarray,
    mask_image: np.ndarray | None,
    auto_mask_mode: str,
    use_auto_mask: bool,
    motion: str,
    duration: float,
    fps: int,
    amplitude: float,
    feather: int,
    output_format: str,
) -> str | None:
    if image is None:
        return None

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if use_auto_mask:
        if auto_mask_mode not in AUTO_MASKS:
            return None
        mask_raw = AUTO_MASKS[auto_mask_mode](bgr)
    elif mask_image is not None:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY) if mask_image.ndim == 3 else mask_image
        mask_raw = mask_gray
    else:
        return None

    amp = amplitude if amplitude > 0 else None
    frames = render_frames(bgr, mask_raw, motion, duration, fps, amp, feather)

    ext = ".gif" if output_format == "GIF" else ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    out_path = tmp.name
    tmp.close()

    if ext == ".gif":
        _write_gif(frames, out_path, fps)
    else:
        h, w = bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

    return out_path


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Cinemagraph Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Cinemagraph Generator\nTurn a static image into a looping motion video.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Input Image", type="numpy")
                mask_input = gr.Image(label="Mask (white = animate)", type="numpy")

                use_auto = gr.Checkbox(label="Use auto-mask instead", value=False)
                auto_mode = gr.Dropdown(
                    choices=list(AUTO_MASKS.keys()),
                    value="water",
                    label="Auto-mask mode",
                    visible=False,
                )
                preview_btn = gr.Button("Preview auto-mask", visible=False)
                mask_preview = gr.Image(label="Auto-mask preview", visible=False, interactive=False)

                motion = gr.Dropdown(choices=list(MOTIONS.keys()), value="ripple", label="Motion preset")
                duration = gr.Slider(0.5, 4.0, value=1.0, step=0.25, label="Duration (seconds)")
                fps_slider = gr.Slider(10, 30, value=24, step=1, label="FPS")
                amplitude = gr.Slider(0, 20, value=0, step=0.5, label="Amplitude (0 = preset default)")
                feather = gr.Slider(0, 20, value=5, step=1, label="Mask feather radius")
                out_format = gr.Radio(["MP4", "GIF"], value="MP4", label="Output format")

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Result")
                download_file = gr.File(label="Download")

        def toggle_auto(checked):
            return (
                gr.update(visible=checked),
                gr.update(visible=checked),
                gr.update(visible=checked),
            )

        use_auto.change(toggle_auto, [use_auto], [auto_mode, preview_btn, mask_preview])
        preview_btn.click(preview_auto_mask, [image_input, auto_mode], mask_preview)

        def on_generate(img, mask_img, auto_mode, use_auto, mot, dur, fps_val, amp, feath, fmt):
            path = run_cinemagraph(img, mask_img, auto_mode, use_auto, mot, dur, int(fps_val), amp, int(feath), fmt)
            if path is None:
                return None, None
            return path, path

        generate_btn.click(
            on_generate,
            [image_input, mask_input, auto_mode, use_auto, motion, duration, fps_slider, amplitude, feather, out_format],
            [output_video, download_file],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
