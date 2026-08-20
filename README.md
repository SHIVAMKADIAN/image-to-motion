# image-to-motion

Two tools to animate static images:

1. **`depth_motion.py`** — Depth-based 3D parallax (Depth Anything V2 Small + OpenCV)
2. **`cinemagraph.py`** — Mask-based cinemagraph loops (OpenCV only)
3. **`frontend/`** — Browser-based UI for Netlify deployment

---

## Depth-Based Parallax (`depth_motion.py`)

Uses monocular depth estimation to create 3D parallax motion — closer objects move more than distant ones.

```bash
pip install -r requirements.txt

# With Depth Anything V2 Small (best quality)
python depth_motion.py photo.jpg -o parallax.mp4 --mode dolly

# Without ML model (pure OpenCV gradient-based depth)
python depth_motion.py photo.jpg -o parallax.mp4 --mode orbit --depth-method gradient

# Save the depth map visualization
python depth_motion.py photo.jpg --save-depth --mode pan --amplitude 15
```

### Depth estimation methods

| Method     | Quality   | Requirements                           |
|------------|-----------|----------------------------------------|
| `auto`     | Best available | Tries ML, falls back to gradient  |
| `ml`       | High      | `transformers`, `torch`, `Pillow`      |
| `gradient` | Decent    | OpenCV + NumPy only (no ML needed)     |

### Parallax modes

| Mode      | Effect                                | Best for                    |
|-----------|---------------------------------------|-----------------------------|
| `dolly`   | Camera push-in/pull-out               | Portraits, landscapes       |
| `pan`     | Horizontal camera slide               | Wide scenes, cityscapes     |
| `tilt`    | Vertical camera slide                 | Tall buildings, waterfalls  |
| `orbit`   | Circular camera movement              | Any photo (most dramatic)   |
| `breathe` | Depth-based pulsing from center       | Portraits, close-ups        |

### CLI options

```
python depth_motion.py <image> [options]

  -o, --output         Output MP4 path (default: parallax.mp4)
  --mode               dolly|pan|tilt|orbit|breathe (default: dolly)
  --duration           Video duration in seconds (default: 2.0)
  --fps                Frames per second (default: 24)
  --amplitude          Motion strength (default: 12.0)
  --depth-method       auto|ml|gradient (default: auto)
  --save-depth         Save depth map as PNG
```

### How it works

1. Load image, resize if > 1024px
2. Estimate depth map (Depth Anything V2 Small or gradient fallback)
3. For each frame, compute displacement proportional to depth × motion function
4. Remap pixels with bilinear interpolation (`cv2.remap`)
5. All motion functions are periodic (sine-based) for seamless looping
6. Export as MP4 via imageio/ffmpeg or OpenCV fallback

### Stack

- **Depth Anything V2 Small** — monocular depth estimation (HuggingFace transformers)
- **OpenCV** — pixel remapping, image I/O
- **NumPy** — frame calculations
- **imageio + PyAV** — H.264 MP4 export (falls back to OpenCV mp4v)

---

## Cinemagraph Generator (`cinemagraph.py`)

Mask-based approach — paint or auto-detect which region to animate.

```bash
# With a hand-drawn mask
python cinemagraph.py photo.jpg mask.png -o cinemagraph.mp4 --motion ripple

# With auto-detected water mask
python cinemagraph.py lake.jpg --auto-mask water --motion flow -o lake.mp4

# Export as GIF
python cinemagraph.py sky.jpg --auto-mask sky --motion cloud -o sky.gif
```

### Motion presets (8 total)

| Preset    | Best for                          |
|-----------|-----------------------------------|
| `ripple`  | Water, puddles, reflections       |
| `flow`    | Rivers, streams, waterfalls       |
| `breathe` | Foliage, fabric                  |
| `cloud`   | Sky, smoke, fog                  |
| `sway`    | Trees, grass                     |
| `shimmer` | Heat haze, glitter               |
| `zoom`    | Pulsating glow                   |
| `spiral`  | Whirlpool, vortex                |

### Auto-mask modes

| Mode    | Detects                        |
|---------|--------------------------------|
| `sky`   | Blue sky + bright overcast     |
| `water` | Blue/cyan water regions        |
| `green` | Foliage and vegetation         |

---

## Web UI (Netlify)

Static frontend — runs entirely in the browser, no server needed.

```bash
# Local preview
cd frontend && python -m http.server 8000
```

### Deploy to Netlify

1. Connect the repo at [app.netlify.com](https://app.netlify.com)
2. `netlify.toml` is pre-configured: publish directory = `frontend/`, no build step
3. Deploy

### Features

- Upload image, paint mask or auto-detect (sky/water/foliage)
- 8 motion presets with amplitude, duration, FPS, feather controls
- Real-time canvas preview
- Export as GIF or WebM
