/**
 * Cinemagraph displacement engine — runs entirely in the browser via Canvas.
 */

const CinemagraphEngine = (() => {

  function ripple(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const cx = w / 2, cy = h / 2;
    const freq = 30;
    const envelope = amp * Math.sin(t);
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-6;
        const phase = (dist / freq) * 2 * Math.PI - t;
        const cosP = Math.cos(phase);
        dx[i] = envelope * cosP * (x - cx) / dist;
        dy[i] = envelope * cosP * (y - cy) / dist;
      }
    }
    return { dx, dy };
  }

  function flow(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const waveLen = 60;
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    const dyVal = amp * 0.3 * Math.sin(t);
    for (let y = 0; y < h; y++) {
      const dxVal = amp * Math.sin((y / waveLen) * 2 * Math.PI + t);
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        dx[i] = dxVal;
        dy[i] = dyVal;
      }
    }
    return { dx, dy };
  }

  function breathe(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const cx = w / 2, cy = h / 2;
    const maxDist = Math.sqrt(cx * cx + cy * cy) + 1e-6;
    const scale = amp * Math.sin(t);
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-6;
        const factor = scale * (dist / maxDist);
        dx[i] = factor * (x - cx) / dist;
        dy[i] = factor * (y - cy) / dist;
      }
    }
    return { dx, dy };
  }

  function cloud(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        dx[i] = amp * Math.sin(t) + amp * 0.3 * Math.sin(y / 40 + t * 2);
        dy[i] = amp * 0.4 * Math.sin(x / 50 + t * 1.5) + amp * 0.2 * Math.cos(y / 35 + t);
      }
    }
    return { dx, dy };
  }

  function sway(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      const yFactor = (y / h) ** 2;
      const dxVal = amp * Math.sin(t) * yFactor;
      const dyVal = amp * 0.15 * Math.sin(t * 2) * yFactor;
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        dx[i] = dxVal;
        dy[i] = dyVal;
      }
    }
    return { dx, dy };
  }

  function shimmer(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        dx[i] = amp * Math.sin(x / 8 + t * 3) * Math.cos(y / 10 + t * 2);
        dy[i] = amp * Math.cos(x / 10 + t * 2.5) * Math.sin(y / 8 + t * 3.5);
      }
    }
    return { dx, dy };
  }

  function zoom(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const cx = w / 2, cy = h / 2;
    const scale = amp * Math.sin(t) / Math.max(h, w);
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        dx[i] = scale * (x - cx);
        dy[i] = scale * (y - cy);
      }
    }
    return { dx, dy };
  }

  function spiral(h, w, frame, totalFrames, amp) {
    const t = (frame / totalFrames) * 2 * Math.PI;
    const cx = w / 2, cy = h / 2;
    const maxDist = Math.sqrt(cx * cx + cy * cy);
    const dx = new Float32Array(h * w);
    const dy = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const rx = x - cx, ry = y - cy;
        const dist = Math.sqrt(rx * rx + ry * ry) + 1e-6;
        const angle = amp * Math.sin(t) * (1 - dist / maxDist) / maxDist;
        dx[i] = rx * Math.cos(angle) - ry * Math.sin(angle) - rx;
        dy[i] = rx * Math.sin(angle) + ry * Math.cos(angle) - ry;
      }
    }
    return { dx, dy };
  }

  const motions = { ripple, flow, breathe, cloud, sway, shimmer, zoom, spiral };

  /**
   * Remap source pixels using displacement, blended by mask.
   * srcData/outData: Uint8ClampedArray (RGBA), mask: Float32Array [0..1], dx/dy: Float32Array
   */
  function remapFrame(srcData, w, h, mask, dxMap, dyMap, outData) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const m = mask[i];
        const pi = i * 4;

        if (m < 0.001) {
          outData[pi] = srcData[pi];
          outData[pi + 1] = srcData[pi + 1];
          outData[pi + 2] = srcData[pi + 2];
          outData[pi + 3] = srcData[pi + 3];
          continue;
        }

        let sx = x + dxMap[i];
        let sy = y + dyMap[i];

        // Reflect at borders
        if (sx < 0) sx = -sx;
        if (sy < 0) sy = -sy;
        if (sx >= w) sx = 2 * w - sx - 2;
        if (sy >= h) sy = 2 * h - sy - 2;
        sx = Math.max(0, Math.min(w - 1.001, sx));
        sy = Math.max(0, Math.min(h - 1.001, sy));

        // Bilinear interpolation
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
        const fx = sx - x0, fy = sy - y0;

        const i00 = (y0 * w + x0) * 4;
        const i10 = (y0 * w + x1) * 4;
        const i01 = (y1 * w + x0) * 4;
        const i11 = (y1 * w + x1) * 4;

        for (let c = 0; c < 3; c++) {
          const top = srcData[i00 + c] * (1 - fx) + srcData[i10 + c] * fx;
          const bot = srcData[i01 + c] * (1 - fx) + srcData[i11 + c] * fx;
          const warped = top * (1 - fy) + bot * fy;
          outData[pi + c] = warped * m + srcData[pi + c] * (1 - m);
        }
        outData[pi + 3] = 255;
      }
    }
  }

  /**
   * Generate all frames as ImageData arrays.
   * Returns array of { imageData } objects.
   */
  function generateFrames(sourceImageData, w, h, maskFloat, motionName, amplitude, duration, fps, feather, onProgress) {
    const totalFrames = Math.round(duration * fps);
    const motionFn = motions[motionName] || motions.ripple;

    // Apply Gaussian-like blur to mask for feathering
    const blurredMask = featherMask(maskFloat, w, h, feather);

    const srcData = sourceImageData.data;
    const frames = [];

    for (let f = 0; f < totalFrames; f++) {
      const { dx, dy } = motionFn(h, w, f, totalFrames, amplitude);
      const outData = new Uint8ClampedArray(srcData.length);
      remapFrame(srcData, w, h, blurredMask, dx, dy, outData);
      frames.push(new ImageData(outData, w, h));
      if (onProgress) onProgress((f + 1) / totalFrames);
    }

    return frames;
  }

  /** Box-blur approximation of Gaussian for mask feathering. */
  function featherMask(mask, w, h, radius) {
    if (radius <= 0) return mask;
    let src = new Float32Array(mask);
    let dst = new Float32Array(mask.length);
    // 3-pass box blur ≈ Gaussian
    for (let pass = 0; pass < 3; pass++) {
      // Horizontal
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let sum = 0, count = 0;
          const x0 = Math.max(0, x - radius);
          const x1 = Math.min(w - 1, x + radius);
          for (let xx = x0; xx <= x1; xx++) {
            sum += src[y * w + xx];
            count++;
          }
          dst[y * w + x] = sum / count;
        }
      }
      [src, dst] = [dst, src];
      // Vertical
      for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
          let sum = 0, count = 0;
          const y0 = Math.max(0, y - radius);
          const y1 = Math.min(h - 1, y + radius);
          for (let yy = y0; yy <= y1; yy++) {
            sum += src[yy * w + x];
            count++;
          }
          dst[y * w + x] = sum / count;
        }
      }
      [src, dst] = [dst, src];
    }
    return src;
  }

  // Auto-mask functions
  function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    let h = 0, s = max === 0 ? 0 : d / max, v = max;
    if (d > 0) {
      if (max === r) h = ((g - b) / d + 6) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60;
    }
    return [h, s * 100, v * 100];
  }

  function autoMaskSky(imageData, w, h) {
    const d = imageData.data;
    const mask = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const posBias = 1 - y / h;
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const [hue, sat, val] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
        const isBlue = hue >= 180 && hue <= 260 && sat > 8 && val > 45;
        const isBright = sat < 25 && val > 70;
        mask[y * w + x] = (isBlue || isBright) ? posBias : 0;
      }
    }
    return mask;
  }

  function autoMaskWater(imageData, w, h) {
    const d = imageData.data;
    const mask = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const posBias = y / h;
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const [hue, sat, val] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
        const isWater = hue >= 170 && hue <= 260 && sat > 12 && val > 15;
        mask[y * w + x] = isWater ? posBias : 0;
      }
    }
    return mask;
  }

  function autoMaskGreen(imageData, w, h) {
    const d = imageData.data;
    const mask = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const [hue, sat, val] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
        const isGreen = hue >= 60 && hue <= 170 && sat > 12 && val > 12;
        mask[y * w + x] = isGreen ? 1 : 0;
      }
    }
    return mask;
  }

  const autoMasks = { sky: autoMaskSky, water: autoMaskWater, green: autoMaskGreen };

  return { motions, generateFrames, autoMasks, featherMask };
})();
