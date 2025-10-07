# Advanced Image Converter — Technical Details

This document provides an in‑depth technical overview of the project implementation, key algorithms, UI architecture, state management, and performance considerations, based on the current code in `main.py`.

## 1. Project Overview
- Desktop GUI application using Tkinter for interactive image stylization.
- Effects supported: Pencil Sketch, Cartoonify (Pastel and Enhance Original have been removed).
- Modes:
  - Single image selection
  - Batch folder processing with navigation and Save All
  - Live camera preview with real‑time effect application
- Dependencies: `opencv-python`, `Pillow`, `numpy` (in `requirements.txt`).

## 2. Code Structure and Key Symbols
- Entry module: `main.py`
- Core state (globals):
  - `current_image: np.ndarray | None` — currently displayed/processed image
  - `current_effect: str | None` — selected effect ("Pencil Sketch" | "Cartoonify")
  - `cap: cv2.VideoCapture | None` — camera capture handle
  - `is_camera_open: bool` — camera mode flag
  - Batch: `batch_images: list[str]`, `batch_index: int`, `batch_mode: bool`
  - UI: `result_label` (display area), `batch_nav_frame` (navigation bar)
- Important functions:
  - `resize_image(img)` — resizes to fit `result_label` while keeping aspect ratio
  - `downscale_for_processing(img, max_dim=800)` — scales large images down for speed
  - `schedule_apply_effect(delay=150)` — debounce for slider changes
  - Effects: `pencil_sketch(img, ...)`, `cartoonify_image(img, ...)`
  - Batch: `open_file(effect, batch=False)`, `load_batch_image(i)`, `next_batch_image()`, `previous_batch_image()`
  - Navigation: `show_batch_navigation()`, `exit_batch_mode()`
  - Display: `display_result(output)`
  - Camera: `open_camera(effect)` (uses `root.after` loop for ~30 FPS)
  - Apply: `apply_effect()` — recompute on slider/effect changes
  - UI bootstrap: `main()` — builds the entire Tkinter UI

## 3. Image Processing Pipelines

### 3.1 Pencil Sketch
- Steps:
  1. Optional downscale via `downscale_for_processing`
  2. Convert to grayscale
  3. Median blur to denoise without destroying edges
  4. Histogram equalization for contrast
  5. Invert + Gaussian blur → smooth shading
  6. Dodge blend: `sketch = 255 * gray / (255 - blurred + eps)`
  7. ConvertScaleAbs for brightness/contrast tuning
  8. Optional sharpening when `sharpness > 1.0` via a custom kernel `_SHARP_KERNEL`
  9. Resize back to original if downscaled
- Parameters used: `brightness`, `contrast`, `sharpness`, `noise_reduction` (+ `hue`, `saturation` ignored for gray)

### 3.2 Cartoonify
- Steps:
  1. Downscale for speed
  2. Denoise with bilateral/edge-preserving filtering to flatten color regions
  3. Convert to grayscale, detect edges via Canny or adaptive threshold
  4. Color quantization (e.g., cluster/levels) to reduce color palette
  5. Combine flattened color with emphasized edges
  6. HSV adjustments (hue shift, saturation, brightness) per sliders
  7. Optional sharpening via `_SHARP_KERNEL`
  8. Resize back to original if needed
- Parameters used: all sliders can affect look, with `hue/saturation/brightness` handled in HSV

## 4. UI Architecture (Tkinter)
- Layout uses a responsive grid:
  - `main_frame` contains a title, control panel, image area, sliders, and `batch_nav_frame`.
  - `result_label` (Label) displays the processed image using a `PhotoImage` created from a PIL Image.
  - Controls:
    - Combobox `effect_dropdown` with values: ["Pencil Sketch", "Cartoonify"]
    - Method selection (e.g., Camera, Open File, Open Folder/Batch)
    - Action buttons: Save, Close/Exit batch, etc.
  - Slider frame includes six sliders: brightness, contrast, saturation, sharpness, hue, noise_reduction.
  - `schedule_apply_effect` is invoked on each slider move to debounce reprocessing.

## 5. Batch Processing and Navigation
- `open_file(effect, batch=True)` loads all images from a user-selected folder into `batch_images` and sets `batch_mode=True`.
- First image is displayed via `load_batch_image(0)`.
- `show_batch_navigation()` renders navigation controls inside `batch_nav_frame`:
  - Buttons: ◀ Previous, Next ▶, Save All, Exit Batch
  - Counter: "Image i of N"
- `next_batch_image()` and `previous_batch_image()` update `batch_index` and call `load_batch_image(index)`.
- `save_all_batch_images()` applies the currently selected effect with the current slider values to all `batch_images` and writes to a selected output directory.
- `exit_batch_mode()` clears state, hides and empties `batch_nav_frame`, resets the result label, and notifies the user.

## 6. Camera Mode
- Opened via `open_camera(effect)` using `cv2.VideoCapture` (CAP_DSHOW on Windows).
- A loop with `root.after` grabs frames, applies the current effect with current slider params, resizes, and displays them.
- Save operation freezes the current frame, re-applies the effect, then opens a file dialog to write the result.

## 7. Display and Resizing
- `resize_image(img)` computes an image size that fits inside the current `result_label` dimensions, with padding and minimum size safeguards.
- Images are converted from BGR (OpenCV) to RGB before converting to PIL and then to a Tkinter-compatible image for display.

## 8. State Management and Events
- Central variables:
  - Image: `current_image` and `current_effect`
  - Mode: `batch_mode`, `batch_images`, `batch_index`, `is_camera_open`
- Event triggers:
  - Combobox change → updates `current_effect` and calls `apply_effect`
  - Slider change → `schedule_apply_effect`
  - File/Folder selection → sets image/batch state and calls `load_batch_image`
  - Camera open → starts display loop; close/exit cleans up resources

## 9. Performance Considerations
- Processing performed on downscaled versions for speed; upscaled back for display/consistency.
- Debouncing via `schedule_apply_effect(delay=150)` avoids redundant recomputation while the user drags sliders.
- Use of efficient OpenCV ops (median/Gaussian blur, Canny, convertScaleAbs) keeps UI responsive.

## 10. Error Handling & UX Details
- Missing file or camera frame → safely continue or show warnings.
- Batch mode gives user feedback (info dialog on load; navigation counter).
- Exit Batch hides and destroys nav widgets to keep UI clean.

## 11. Build and Run
- Requirements:
  - Python 3.8+
  - `pip install -r requirements.txt`
- Run:
  - `python main.py`
- Notes for Windows camera:
  - Uses `cv2.CAP_DSHOW` which often improves compatibility.

## 12. Limitations & Future Enhancements
- Effects limited to two styles; plugins or additional presets can be added.
- No GPU acceleration; potential speed improvements via OpenCL/CUDA if available.
- Advanced color quantization (e.g., k-means on downscaled image) can further improve Cartoonify quality.
- Persist user preferences across sessions; add drag‑and‑drop for images/folders.
- Add unit tests for processing functions using small sample images to guard regressions.

## 13. Data Flow Summary
1. Input source chosen (single/batch/camera)
2. Image/frame loaded into `current_image`
3. User selects effect and adjusts sliders
4. `apply_effect` computes processed output using selected pipeline
5. `resize_image` adapts result to display area
6. `display_result` renders to `result_label`
7. User saves current image or all batch outputs

## 14. Key UI Elements (IDs/Variables)
- `result_label`: image display surface
- `effect_dropdown`: Combobox of effects
- `batch_nav_frame`: container for navigation buttons and counter
- Sliders bound to: `brightness_val`, `contrast_val`, `saturation_val`, `sharpness_val`, `hue_val`, `noise_reduction_val`

## 15. Security & Privacy
- No network or cloud operations; all processing occurs locally.
- No persistence of images beyond user-initiated saves.

---
If you need diagrams (architecture or pipeline blocks), we can add Mermaid diagrams to this document in a follow-up.
