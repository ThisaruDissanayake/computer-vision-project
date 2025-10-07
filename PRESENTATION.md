# Advanced Image Converter – Presentation

## Slide 1 — Title & Overview
- Advanced Image Converter
- A Tkinter + OpenCV desktop app to stylize images (Pencil Sketch, Cartoonify)
- Modes: Single image, Batch folder, Live Camera
- Responsive UI with real‑time sliders and batch navigation

## Slide 2 — Problem & Goal
- Problem: Quickly preview and export stylized images without complex tools
- Goal: Provide a lightweight, fast, and interactive GUI for cartoon/sketch effects
- Constraints: Desktop‑friendly, minimal dependencies, smooth UX

## Slide 3 — Key Features
- Two effects: Pencil Sketch and Cartoonify
- Real‑time preview for camera and images
- Batch processing with Previous/Next navigation and Save All
- Adjustable sliders: brightness, contrast, saturation, sharpness, hue, noise reduction
- Clean, responsive layout (grid) and history window

## Slide 4 — System Architecture
- Frontend: Tkinter (Frames, Labels, Scales, Combobox, Buttons)
- Image Core: OpenCV + NumPy
- Data Flow:
  1) Load (file/folder/camera) →
  2) Apply effect (params from sliders) →
  3) Resize for display →
  4) Show on result label →
  5) Save (single/all)
- State: current_image, current_effect, batch_images, batch_index, batch_mode, is_camera_open

## Slide 5 — Effects: Pencil Sketch
- Pipeline:
  - Grayscale → median blur → histogram equalization
  - Invert + Gaussian blur
  - Dodge blend: sketch = 255 * gray / (255 - blur + eps)
  - Optional sharpening; brightness/contrast tweak via convertScaleAbs
- Result: Clean pencil‑style strokes with adjustable clarity

## Slide 6 — Effects: Cartoonify
- Pipeline:
  - Downscale for speed → bilateral/edge‑preserving smoothing
  - Canny/threshold edges → quantize colors (cluster/levels)
  - Blend flattened colors with bold edges
  - HSV adjustments (hue, saturation, brightness) + sharpening optional
- Result: Strong edges and flat color regions for a cartoon look

## Slide 7 — Batch Processing UX
- Select a folder → images enumerated → first image displayed
- Navigation bar: ◀ Previous | Image i of N | Next ▶ | Save All | Exit Batch
- Save All applies current effect and parameters to every image in the batch
- Exit Batch resets UI and hides navigation controls

## Slide 8 — Camera Mode UX
- Opens default camera using CAP_DSHOW on Windows; tries fallbacks
- Live preview at ~30 FPS using root.after scheduling
- Snapshot on Save: freezes current frame, applies selected effect, opens save dialog

## Slide 9 — Performance & Responsiveness
- Downscaling for processing (max dim ~800 px) for speed
- Debounced slider updates via schedule_apply_effect(delay=150 ms)
- Efficient conversions and lazy redraws
- Grid weights for responsive resizing; resized display to fit label bounds

## Slide 10 — Installation, Run & Roadmap
- Install: pip install -r requirements.txt
- Run: python main.py
- Roadmap:
  - Add export presets (PNG/JPEG quality), CLI mode
  - More effects (watercolor, manga), GPU acceleration when available
  - Persist user settings; drag‑and‑drop for images/folders
