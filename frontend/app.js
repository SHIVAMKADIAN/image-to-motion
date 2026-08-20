/**
 * Cinemagraph UI — wires up the HTML to the engine.
 */

(function () {
  // State
  let sourceImage = null;   // HTMLImageElement
  let sourceData = null;     // ImageData
  let imgW = 0, imgH = 0;
  let maskData = null;       // Float32Array [0..1]
  let generatedFrames = [];  // ImageData[]
  let playInterval = null;
  let playIndex = 0;
  let isPlaying = false;
  let paintMode = "paint";   // "paint" | "erase"

  // Max canvas dimension to keep performance reasonable
  const MAX_DIM = 800;

  // Elements
  const fileInput = document.getElementById("file-input");
  const uploadZone = document.getElementById("upload-zone");
  const previewImg = document.getElementById("preview-img");
  const canvasMain = document.getElementById("canvas-main");
  const canvasMask = document.getElementById("canvas-mask");
  const canvasPreview = document.getElementById("canvas-preview");
  const ctxMain = canvasMain.getContext("2d");
  const ctxMask = canvasMask.getContext("2d");
  const ctxPreview = canvasPreview.getContext("2d");

  const stepMask = document.getElementById("step-mask");
  const stepMotion = document.getElementById("step-motion");
  const stepPreview = document.getElementById("step-preview");

  const brushSize = document.getElementById("brush-size");
  const brushSizeVal = document.getElementById("brush-size-val");
  const btnPaint = document.getElementById("btn-paint");
  const btnErase = document.getElementById("btn-erase");
  const btnClear = document.getElementById("btn-clear-mask");
  const paintControls = document.getElementById("paint-controls");
  const autoControls = document.getElementById("auto-controls");
  const autoThreshold = document.getElementById("auto-threshold");
  const autoThresholdVal = document.getElementById("auto-threshold-val");

  const paramAmplitude = document.getElementById("param-amplitude");
  const paramDuration = document.getElementById("param-duration");
  const paramFps = document.getElementById("param-fps");
  const paramFeather = document.getElementById("param-feather");
  const paramAmplitudeVal = document.getElementById("param-amplitude-val");
  const paramDurationVal = document.getElementById("param-duration-val");
  const paramFpsVal = document.getElementById("param-fps-val");
  const paramFeatherVal = document.getElementById("param-feather-val");

  const btnGenerate = document.getElementById("btn-generate");
  const btnPlay = document.getElementById("btn-play");
  const progressBar = document.getElementById("progress-bar");
  const progressFill = document.getElementById("progress-fill");
  const progressText = document.getElementById("progress-text");
  const exportButtons = document.getElementById("export-buttons");
  const btnExportGif = document.getElementById("btn-export-gif");
  const btnExportWebm = document.getElementById("btn-export-webm");

  // ── Upload ──────────────────────────────────────────────────
  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) loadImage(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) loadImage(fileInput.files[0]);
  });

  function loadImage(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      sourceImage = img;

      // Scale down if too large
      let w = img.naturalWidth, h = img.naturalHeight;
      if (w > MAX_DIM || h > MAX_DIM) {
        const scale = MAX_DIM / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      imgW = w;
      imgH = h;

      // Show preview
      previewImg.src = url;
      previewImg.hidden = false;
      uploadZone.classList.add("has-image");
      uploadZone.querySelector(".upload-prompt").hidden = true;

      // Set up canvases
      canvasMain.width = w;
      canvasMain.height = h;
      ctxMain.drawImage(img, 0, 0, w, h);
      sourceData = ctxMain.getImageData(0, 0, w, h);

      canvasMask.width = w;
      canvasMask.height = h;
      maskData = new Float32Array(w * h);
      drawMaskOverlay();

      canvasPreview.width = w;
      canvasPreview.height = h;
      ctxPreview.drawImage(img, 0, 0, w, h);

      // Enable steps
      stepMask.classList.remove("disabled");
      stepMotion.classList.remove("disabled");
      stepPreview.classList.remove("disabled");

      // Reset
      generatedFrames = [];
      stopPlayback();
      exportButtons.hidden = true;
      btnPlay.hidden = true;
    };
    img.src = url;
  }

  // ── Mask painting ──────────────────────────────────────────
  let isPainting = false;

  function getCanvasPos(e) {
    const rect = canvasMask.getBoundingClientRect();
    const scaleX = imgW / rect.width;
    const scaleY = imgH / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  function paintAt(x, y) {
    if (!maskData) return;
    const r = parseInt(brushSize.value);
    const val = paintMode === "paint" ? 1 : 0;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r * r) continue;
        const px = Math.round(x + dx), py = Math.round(y + dy);
        if (px < 0 || px >= imgW || py < 0 || py >= imgH) continue;
        maskData[py * imgW + px] = val;
      }
    }
    drawMaskOverlay();
  }

  function onPointerDown(e) {
    e.preventDefault();
    isPainting = true;
    const pos = getCanvasPos(e);
    paintAt(pos.x, pos.y);
  }

  function onPointerMove(e) {
    if (!isPainting) return;
    e.preventDefault();
    const pos = getCanvasPos(e);
    paintAt(pos.x, pos.y);
  }

  function onPointerUp() { isPainting = false; }

  canvasMask.addEventListener("mousedown", onPointerDown);
  canvasMask.addEventListener("mousemove", onPointerMove);
  canvasMask.addEventListener("mouseup", onPointerUp);
  canvasMask.addEventListener("mouseleave", onPointerUp);
  canvasMask.addEventListener("touchstart", onPointerDown, { passive: false });
  canvasMask.addEventListener("touchmove", onPointerMove, { passive: false });
  canvasMask.addEventListener("touchend", onPointerUp);

  function drawMaskOverlay() {
    ctxMask.clearRect(0, 0, imgW, imgH);
    if (!maskData) return;
    const overlay = ctxMask.createImageData(imgW, imgH);
    for (let i = 0; i < maskData.length; i++) {
      const m = maskData[i];
      overlay.data[i * 4] = 100;       // R
      overlay.data[i * 4 + 1] = 200;   // G
      overlay.data[i * 4 + 2] = 255;   // B
      overlay.data[i * 4 + 3] = m * 120; // A
    }
    ctxMask.putImageData(overlay, 0, 0);
  }

  // Paint/erase buttons
  btnPaint.addEventListener("click", () => {
    paintMode = "paint";
    btnPaint.classList.add("active");
    btnErase.classList.remove("active");
  });
  btnErase.addEventListener("click", () => {
    paintMode = "erase";
    btnErase.classList.add("active");
    btnPaint.classList.remove("active");
  });
  btnClear.addEventListener("click", () => {
    if (!maskData) return;
    maskData.fill(0);
    drawMaskOverlay();
  });

  brushSize.addEventListener("input", () => {
    brushSizeVal.textContent = brushSize.value;
  });

  // ── Mask mode tabs ─────────────────────────────────────────
  document.querySelectorAll(".mask-mode-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".mask-mode-tabs .tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.dataset.mode;
      paintControls.hidden = mode !== "paint";
      autoControls.hidden = mode !== "auto";
      canvasMask.style.pointerEvents = mode === "paint" ? "auto" : "none";
    });
  });

  // Auto-mask buttons
  document.querySelectorAll("[data-auto]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!sourceData) return;
      const mode = btn.dataset.auto;
      const fn = CinemagraphEngine.autoMasks[mode];
      if (!fn) return;
      maskData = fn(sourceData, imgW, imgH);
      drawMaskOverlay();
    });
  });

  autoThreshold.addEventListener("input", () => {
    autoThresholdVal.textContent = autoThreshold.value;
  });

  // ── Motion preset selection ────────────────────────────────
  let selectedMotion = "ripple";
  document.querySelectorAll(".motion-card").forEach((card) => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".motion-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      selectedMotion = card.dataset.motion;
    });
  });

  // Param display updates
  paramAmplitude.addEventListener("input", () => { paramAmplitudeVal.textContent = paramAmplitude.value; });
  paramDuration.addEventListener("input", () => { paramDurationVal.textContent = paramDuration.value + "s"; });
  paramFps.addEventListener("input", () => { paramFpsVal.textContent = paramFps.value; });
  paramFeather.addEventListener("input", () => { paramFeatherVal.textContent = paramFeather.value; });

  // ── Generate ───────────────────────────────────────────────
  btnGenerate.addEventListener("click", () => {
    if (!sourceData || !maskData) return;

    const hasMask = maskData.some((v) => v > 0.01);
    if (!hasMask) {
      alert("Please paint or auto-detect a mask area first.");
      return;
    }

    stopPlayback();
    btnGenerate.disabled = true;
    btnGenerate.textContent = "Generating...";
    progressBar.hidden = false;
    exportButtons.hidden = true;
    btnPlay.hidden = true;

    const amplitude = parseFloat(paramAmplitude.value);
    const duration = parseFloat(paramDuration.value);
    const fps = parseInt(paramFps.value);
    const feather = parseInt(paramFeather.value);

    // Run in chunks to keep UI responsive
    setTimeout(() => {
      generatedFrames = CinemagraphEngine.generateFrames(
        sourceData, imgW, imgH, maskData,
        selectedMotion, amplitude, duration, fps, feather,
        (progress) => {
          const pct = Math.round(progress * 100);
          progressFill.style.width = pct + "%";
          progressText.textContent = pct + "%";
        }
      );

      btnGenerate.disabled = false;
      btnGenerate.textContent = "Generate Animation";
      progressBar.hidden = true;
      exportButtons.hidden = false;
      btnPlay.hidden = false;

      // Start playback
      startPlayback(fps);
    }, 50);
  });

  // ── Playback ───────────────────────────────────────────────
  function startPlayback(fps) {
    stopPlayback();
    if (!generatedFrames.length) return;
    isPlaying = true;
    playIndex = 0;
    btnPlay.textContent = "Pause";
    const interval = 1000 / (fps || 20);
    playInterval = setInterval(() => {
      if (!isPlaying || !generatedFrames.length) return;
      ctxPreview.putImageData(generatedFrames[playIndex], 0, 0);
      playIndex = (playIndex + 1) % generatedFrames.length;
    }, interval);
  }

  function stopPlayback() {
    if (playInterval) clearInterval(playInterval);
    playInterval = null;
    isPlaying = false;
    btnPlay.textContent = "Play";
  }

  btnPlay.addEventListener("click", () => {
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback(parseInt(paramFps.value));
    }
  });

  // ── Export GIF ─────────────────────────────────────────────
  btnExportGif.addEventListener("click", () => {
    if (!generatedFrames.length) return;
    btnExportGif.textContent = "Encoding...";
    btnExportGif.disabled = true;

    setTimeout(() => {
      const fps = parseInt(paramFps.value);
      const data = GIFEncoder.encode(generatedFrames, imgW, imgH, fps);
      const blob = new Blob([data], { type: "image/gif" });
      downloadBlob(blob, "cinemagraph.gif");
      btnExportGif.textContent = "Download GIF";
      btnExportGif.disabled = false;
    }, 50);
  });

  // ── Export WebM ────────────────────────────────────────────
  btnExportWebm.addEventListener("click", () => {
    if (!generatedFrames.length) return;
    btnExportWebm.textContent = "Recording...";
    btnExportWebm.disabled = true;

    const fps = parseInt(paramFps.value);
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = imgW;
    tempCanvas.height = imgH;
    const tempCtx = tempCanvas.getContext("2d");

    const stream = tempCanvas.captureStream(0);
    const recorder = new MediaRecorder(stream, {
      mimeType: "video/webm;codecs=vp9",
      videoBitsPerSecond: 5000000,
    });

    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      downloadBlob(blob, "cinemagraph.webm");
      btnExportWebm.textContent = "Download WebM";
      btnExportWebm.disabled = false;
    };

    recorder.start();

    // Write 3 loops for a longer video
    const allFrames = [...generatedFrames, ...generatedFrames, ...generatedFrames];
    let fi = 0;
    const frameInterval = setInterval(() => {
      if (fi >= allFrames.length) {
        clearInterval(frameInterval);
        recorder.stop();
        return;
      }
      tempCtx.putImageData(allFrames[fi], 0, 0);
      stream.getVideoTracks()[0].requestFrame();
      fi++;
    }, 1000 / fps);
  });

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
})();
