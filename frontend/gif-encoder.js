/**
 * Minimal GIF89a encoder for animated GIFs.
 * Supports looping, per-frame delay, and median-cut color quantization.
 */

const GIFEncoder = (() => {

  function encode(frames, w, h, fps) {
    const delay = Math.max(2, Math.round(100 / fps));
    const parts = [];

    // Header
    parts.push(str("GIF89a"));
    parts.push(uint16(w), uint16(h));

    // Quantize first frame for global color table
    const { indices: firstIdx, palette: firstPal } = quantize(frames[0], w, h);
    parts.push(new Uint8Array([0xF7, 0, 0])); // GCT flag, 256 colors
    for (let i = 0; i < 256; i++) {
      if (i < firstPal.length) {
        parts.push(new Uint8Array([firstPal[i][0], firstPal[i][1], firstPal[i][2]]));
      } else {
        parts.push(new Uint8Array([0, 0, 0]));
      }
    }

    // NETSCAPE extension for looping
    parts.push(new Uint8Array([
      0x21, 0xFF, 0x0B,
      0x4E, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, // NETSCAPE
      0x32, 0x2E, 0x30, // 2.0
      0x03, 0x01, 0x00, 0x00, 0x00 // loop forever
    ]));

    for (let f = 0; f < frames.length; f++) {
      const { indices } = quantize(frames[f], w, h);

      // GCE
      parts.push(new Uint8Array([0x21, 0xF9, 0x04, 0x00]));
      parts.push(uint16(delay));
      parts.push(new Uint8Array([0x00, 0x00]));

      // Image descriptor
      parts.push(new Uint8Array([0x2C]));
      parts.push(uint16(0), uint16(0), uint16(w), uint16(h));
      parts.push(new Uint8Array([0x00])); // no local color table

      // LZW
      const minCodeSize = 8;
      parts.push(new Uint8Array([minCodeSize]));
      const compressed = lzwCompress(indices, minCodeSize);
      // Sub-block the data
      let offset = 0;
      while (offset < compressed.length) {
        const chunk = Math.min(255, compressed.length - offset);
        parts.push(new Uint8Array([chunk]));
        parts.push(compressed.subarray(offset, offset + chunk));
        offset += chunk;
      }
      parts.push(new Uint8Array([0x00])); // block terminator
    }

    parts.push(new Uint8Array([0x3B])); // trailer

    const totalLen = parts.reduce((s, p) => s + p.length, 0);
    const result = new Uint8Array(totalLen);
    let pos = 0;
    for (const p of parts) {
      result.set(p, pos);
      pos += p.length;
    }
    return result;
  }

  function quantize(imageData, w, h) {
    const data = imageData.data;
    const n = w * h;

    // K-means-ish: sample pixels, find 256 centroids
    const sampleRate = Math.max(1, Math.floor(n / 4000));
    const samples = [];
    for (let i = 0; i < n; i += sampleRate) {
      const p = i * 4;
      samples.push([data[p], data[p + 1], data[p + 2]]);
    }

    // Simple uniform quantization for speed (5-6-5 bits → remap to 256)
    const palette = [];
    const colorMap = new Map();
    for (const [r, g, b] of samples) {
      const key = ((r >> 3) << 10) | ((g >> 2) << 5) | (b >> 3);
      if (!colorMap.has(key)) {
        colorMap.set(key, [r, g, b]);
      }
    }

    // Take up to 256 representative colors
    const allColors = Array.from(colorMap.values());
    if (allColors.length <= 256) {
      palette.push(...allColors);
    } else {
      const step = allColors.length / 256;
      for (let i = 0; i < 256; i++) {
        palette.push(allColors[Math.floor(i * step)]);
      }
    }
    while (palette.length < 256) palette.push([0, 0, 0]);

    // Build lookup cache
    const cache = new Map();
    const indices = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      const r = data[p], g = data[p + 1], b = data[p + 2];
      const cacheKey = ((r >> 2) << 12) | ((g >> 2) << 6) | (b >> 2);
      if (cache.has(cacheKey)) {
        indices[i] = cache.get(cacheKey);
        continue;
      }
      let bestDist = Infinity, bestIdx = 0;
      for (let j = 0; j < 256; j++) {
        const dr = r - palette[j][0], dg = g - palette[j][1], db = b - palette[j][2];
        const dist = dr * dr + dg * dg + db * db;
        if (dist < bestDist) { bestDist = dist; bestIdx = j; }
      }
      indices[i] = bestIdx;
      cache.set(cacheKey, bestIdx);
    }

    return { indices, palette };
  }

  function lzwCompress(data, minCodeSize) {
    const clearCode = 1 << minCodeSize;
    const eoiCode = clearCode + 1;
    let codeSize = minCodeSize + 1;
    let nextCode = eoiCode + 1;
    let maxCode = (1 << codeSize) - 1;

    const table = new Map();
    for (let i = 0; i < clearCode; i++) table.set(String(i), i);

    let bitBuf = 0, bitCount = 0;
    const result = [];

    function emit(code) {
      bitBuf |= code << bitCount;
      bitCount += codeSize;
      while (bitCount >= 8) {
        result.push(bitBuf & 0xFF);
        bitBuf >>= 8;
        bitCount -= 8;
      }
    }

    emit(clearCode);
    let w = String(data[0]);

    for (let i = 1; i < data.length; i++) {
      const k = String(data[i]);
      const wk = w + "," + k;
      if (table.has(wk)) {
        w = wk;
      } else {
        emit(table.get(w));
        if (nextCode <= 4095) {
          table.set(wk, nextCode++);
          if (nextCode > maxCode + 1 && codeSize < 12) {
            codeSize++;
            maxCode = (1 << codeSize) - 1;
          }
        } else {
          emit(clearCode);
          table.clear();
          for (let j = 0; j < clearCode; j++) table.set(String(j), j);
          nextCode = eoiCode + 1;
          codeSize = minCodeSize + 1;
          maxCode = (1 << codeSize) - 1;
        }
        w = k;
      }
    }
    emit(table.get(w));
    emit(eoiCode);
    if (bitCount > 0) result.push(bitBuf & 0xFF);

    return new Uint8Array(result);
  }

  function str(s) { return new TextEncoder().encode(s); }
  function uint16(v) { return new Uint8Array([v & 0xFF, (v >> 8) & 0xFF]); }

  return { encode };
})();
