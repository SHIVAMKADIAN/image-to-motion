# image-to-motion

Turn a static image into a looping cinemagraph (~1 second of subtle motion).

Provide an image and a grayscale mask (white = regions to animate, black = freeze), pick a motion preset, and get a seamless looping MP4.

## Quick start

```bash
pip install -r requirements.txt
python cinemagraph.py photo.jpg mask.png -o cinemagraph.mp4 --motion ripple
```

## Motion presets

| Preset   | Best for                        |
|----------|---------------------------------|
| `ripple` | Water, puddles, reflections     |
| `flow`   | Rivers, streams, waterfalls     |
| `breathe`| Foliage, fabric, gentle sway    |
| `cloud`  | Sky, smoke, fog                 |

## Options

```
positional:
  image              Input image (jpg/png)
  mask               Grayscale mask — white regions animate

optional:
  -o, --output       Output path (default: output.mp4)
  --motion           ripple | flow | breathe | cloud (default: ripple)
  --duration         Loop length in seconds (default: 1.0)
  --fps              Frames per second (default: 24)
  --amplitude        Override motion strength
  --feather          Mask edge softness radius (default: 5)
```

## How it works

1. Load image + mask
2. Feather the mask edges for smooth blending
3. For each frame, compute a displacement map (sine-based warping)
4. Remap the masked region with the displacement
5. Blend warped region back onto the static background
6. All displacement functions are periodic, so the output loops seamlessly

## Creating masks

Use any image editor (GIMP, Photoshop, Paint) to paint white over the areas you want animated (water, clouds, etc.) and black over everything else. Save as a grayscale PNG.
