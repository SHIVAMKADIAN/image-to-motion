# image-to-motion

Turn a static image into a looping cinemagraph — subtle, seamless motion in a specific region while the rest stays frozen.

## Quick start

```bash
pip install -r requirements.txt

# With a hand-drawn mask
python cinemagraph.py photo.jpg mask.png -o cinemagraph.mp4 --motion ripple

# With auto-detected water mask
python cinemagraph.py lake.jpg --auto-mask water --motion flow -o lake.mp4

# Export as GIF
python cinemagraph.py sky.jpg --auto-mask sky --motion cloud -o sky.gif
```

## Web UI

```bash
python app.py
# Opens at http://localhost:7860
```

Upload an image, draw or auto-detect a mask, pick a motion preset, and download the result.

## Motion presets

| Preset    | Effect                                  | Best for                          |
|-----------|-----------------------------------------|-----------------------------------|
| `ripple`  | Concentric sine-wave displacement       | Water, puddles, reflections       |
| `flow`    | Directional horizontal drift + wobble   | Rivers, streams, waterfalls       |
| `breathe` | Radial expand/contract                 | Foliage, fabric, gentle sway      |
| `cloud`   | Lateral drift with turbulence          | Sky, smoke, fog                   |
| `sway`    | Pendulum swing (top fixed)             | Trees, grass, hanging objects     |
| `shimmer` | High-frequency micro-jitter            | Heat haze, glitter, light on water|
| `zoom`    | Pulsing zoom in/out from center        | Heartbeat, pulsating glow         |
| `spiral`  | Rotational twist around center         | Whirlpool, vortex                 |

## Auto-mask modes

Skip the manual mask — auto-detect regions by color:

| Mode    | Detects                        |
|---------|--------------------------------|
| `sky`   | Blue sky + bright overcast     |
| `water` | Blue/cyan water regions        |
| `green` | Foliage and vegetation         |

```bash
python cinemagraph.py photo.jpg --auto-mask water --motion ripple
```

## CLI options

```
positional:
  image              Input image (jpg/png)
  mask               Grayscale mask (optional if --auto-mask used)

optional:
  -o, --output       Output path — .mp4 or .gif (default: output.mp4)
  --motion           ripple|flow|breathe|cloud|sway|shimmer|zoom|spiral
  --auto-mask        sky|water|green — auto-detect mask region
  --duration         Loop length in seconds (default: 1.0)
  --fps              Frames per second (default: 24)
  --amplitude        Override motion strength (0 = preset default)
  --feather          Mask edge softness radius (default: 5)
```

## How it works

1. Load image + mask (hand-drawn or auto-detected via HSV color analysis)
2. Feather the mask edges with Gaussian blur for smooth blending
3. For each frame, compute a periodic displacement map (sine-based warping)
4. Remap the masked region with `cv2.remap`
5. Alpha-blend warped region back onto the frozen background
6. All displacement functions are periodic over the duration, so the output loops seamlessly
7. Write as MP4 (OpenCV VideoWriter) or GIF (built-in LZW encoder)

## Creating masks manually

Paint white over the areas you want animated (water, clouds, etc.) in any image editor and save as a grayscale PNG. Black = frozen, white = animated, gray = partial blend.
