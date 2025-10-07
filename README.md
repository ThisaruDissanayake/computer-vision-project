# Advanced Image Converter

A small GUI tool to apply stylized image effects (pencil sketch, cartoonify, pastel, and enhancement) to images and live camera feed using OpenCV and Tkinter. The interface includes sliders for brightness, contrast, saturation, sharpness, hue shift, and noise reduction. You can process single images, a batch of images from a directory, or a connected camera.

## Features

- Pencil Sketch, Cartoonify, Pastel, and Enhance Original effects
- Live camera preview with real-time effect application
- Save processed frames or images
- Batch process a directory of images
- History view with thumbnails

## Prerequisites

- Python 3.8+ (3.10 or 3.11 recommended)
- A working camera if you want live camera functionality

## Python dependencies

The project uses the following Python packages:

- opencv-python
- Pillow

A `requirements.txt` is provided for convenience.

## Installation

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip; pip install -r requirements.txt
```

## Running

From the project root run:

```powershell
python main.py
```

This will open a GUI window. Select an effect and a method (Camera / Upload Image / Batch Process). Adjust sliders to tune the effect, then `Run`.

### Notes

- If the camera cannot be opened, ensure no other application is using it and that your system has a working camera driver.
- Batch processing saves processed images via a save dialog for each file.

## Troubleshooting

- On Windows, if OpenCV cannot access the camera, try installing the `opencv-python` package (not `opencv-contrib-python`) and run with the `cv2.CAP_DSHOW` flag (already used in the code).
- If the GUI appears blank, check that `PIL` (Pillow) is installed and that images are read correctly (non-empty `cv2.imread` result).

## Files

- `main.py` - Main application file
- `README.md` - This file
- `requirements.txt` - Python dependencies

## License

This repository contains example code. Add a license if you plan to publish it.

## Techniques used

This project implements several image stylization and enhancement techniques using OpenCV and NumPy, with a Tkinter GUI for interaction. Below is a concise description of the main techniques, design choices, and implementation notes found in `main.py`:

- Pencil Sketch
	- Converts the image to grayscale, applies median blur (configurable via a "noise reduction" slider), and uses histogram equalization to improve contrast.
	- Uses an inverted blurred image and a dodge blend style (fast division) to produce a pencil-sketch look.
	- Optional sharpening is applied with a small convolution kernel when the sharpness slider is increased.

- Cartoonify
	- Performs edge detection on a blurred grayscale image using adaptive thresholding to extract strong edges.
	- Uses edge-preserving filters and bilateral filtering on a downscaled color image to create flattened color regions.
	- Adjusts hue, saturation, and brightness in HSV space for color tuning; then combines color regions with the edge mask.

- Pastel
	- Converts to HSV and reduces saturation while boosting brightness slightly to create a pastel palette.
	- Applies a light Gaussian blur for softness and optional sharpening for detail preservation.

- Enhance Original
	- Adjusts hue/saturation/brightness in HSV and applies contrast/brightness scaling to improve color and exposure.
	- Optional sharpening and light median blur for noise reduction are supported.

- Performance and UI considerations
	- Downscaling for processing: Large images are downscaled to a max dimension (default 800 px) for faster processing, then resized back to original dimensions. This balances performance and quality.
	- Debounced slider updates: Slider changes are debounced (short delay) before re-applying effects to avoid excessive CPU usage while dragging.
	- Fast filters: Uses OpenCV built-in fast filters (medianBlur, GaussianBlur, bilateralFilter, edgePreservingFilter) rather than slower clustering-based methods (e.g., K-Means) for responsive UI.
	- Camera handling: Tries multiple camera indices and uses the `cv2.CAP_DSHOW` backend on Windows to improve reliability.
	- Save/history: Processed frames can be saved via a file dialog; a simple in-memory history stores thumbnails for quick re-saving.

- Parameters exposed in UI
	- Brightness, Contrast, Saturation, Sharpness, Hue Shift, Noise Reduction — all mapped to the underlying OpenCV operations.

These techniques are implemented in `main.py` with the goal of achieving interactive responsiveness on typical consumer hardware while still producing visually pleasing results. If you want higher-quality but slower variants (for example, larger bilateral filter settings, K-Means color quantization, or multi-scale processing), those can be added as optional modes.
