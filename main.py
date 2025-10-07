import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Label, ttk, Scale, Button, Toplevel
from PIL import Image, ImageTk
import numpy as np
import os

# Global variables to store current image, effect, and camera state
current_image = None
current_effect = None
cap = None
is_camera_open = False
_PROCESSING_MAX_DIM = 800
_SHARP_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
# ID for debounced apply_effect scheduling
_apply_after_id = None

# Resize image or frame to fit the 600x400 result area while maintaining aspect ratio
def resize_image(img):
    height, width = img.shape[:2]
    target_width, target_height = 600, 400
    aspect_ratio = width / height
    
    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
        if new_height > target_height:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        if new_width > target_width:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
    
    return cv2.resize(img, (new_width, new_height))


def downscale_for_processing(img, max_dim=_PROCESSING_MAX_DIM):
    """Downscale large images for faster processing, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def schedule_apply_effect(delay=150):
    """Debounce apply_effect calls from frequent slider events."""
    global _apply_after_id
    try:
        if _apply_after_id is not None:
            root.after_cancel(_apply_after_id)
    except Exception:
        pass
    _apply_after_id = root.after(delay, apply_effect)

# Enhanced pencil sketch effect with fast processing
def pencil_sketch(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, hue=0.0, noise_reduction=0.0):
    # Downscale for faster processing
    small = downscale_for_processing(image)
    
    if len(small.shape) == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = small
    
    ksize = int(5 + noise_reduction * 4)
    ksize = max(3, ksize + (1 - ksize % 2))
    gray = cv2.medianBlur(gray, ksize)
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=(brightness - 1.0) * 100)
    gray = cv2.equalizeHist(gray)

    # Fast dodge blend
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
    
    if sharpness > 1.0:
        sharpened_kernel = _SHARP_KERNEL * sharpness
        sketch = cv2.filter2D(sketch, -1, sharpened_kernel)
    
    # Resize back to original dimensions if downscaled
    if small.shape[:2] != image.shape[:2]:
        sketch = cv2.resize(sketch, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return sketch

# Enhanced cartoonify effect with fast processing
def cartoonify_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, hue=0.0, noise_reduction=0.0):
    # Downscale for faster processing
    small = downscale_for_processing(image)
    
    # Fast edge detection
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    # Fast color smoothing (replace expensive KMeans)
    color = cv2.edgePreservingFilter(small, flags=1, sigma_s=60, sigma_r=0.4)
    color = cv2.bilateralFilter(color, d=9, sigmaColor=150 + noise_reduction * 50, sigmaSpace=150)
    
    # Apply adjustments
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue_shift = int(hue * 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation * 1.2, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness * 1.1, 0, 255)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Sharpen if needed
    if sharpness > 1.0:
        sharpened_kernel = _SHARP_KERNEL * sharpness
        color = cv2.filter2D(color, -1, sharpened_kernel)
    
    # Combine with edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)
    cartoon = cv2.convertScaleAbs(cartoon, alpha=contrast * 1.3, beta=(brightness - 1.0) * 10)
    
    # Resize back to original dimensions if downscaled
    if small.shape[:2] != image.shape[:2]:
        cartoon = cv2.resize(cartoon, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return cartoon

# Pastel color effect with fast processing
def pastel_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, hue=0.0, noise_reduction=0.0):
    # Downscale for faster processing
    small = downscale_for_processing(image)

    # Convert to float for smoother adjustments
    img_f = small.astype(np.float32) / 255.0

    # Slight hue shift in HSV
    hsv = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = int(hue * 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

    # Reduce saturation gently to get pastel look, but allow user control
    hsv[:, :, 1] = hsv[:, :, 1] * (0.6 + 0.4 * saturation)

    # Boost brightness slightly for pastel tones
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + 0.15 * (brightness - 1.0)), 0, 255)

    pastel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Edge-preserving smoothing to create soft color regions while preserving edges
    pastel = cv2.edgePreservingFilter(pastel, flags=1, sigma_s=60, sigma_r=0.4)

    # Gentle bilateral filtering for painterly softness; scale with noise_reduction
    d = 5
    sigma_color = 75 + noise_reduction * 50
    sigma_space = 75
    pastel = cv2.bilateralFilter(pastel, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Apply mild gamma correction / tone mapping using contrast param
    gamma = 1.0 / max(0.01, contrast)
    pastel_f = pastel.astype(np.float32) / 255.0
    pastel_f = np.clip(pastel_f ** gamma, 0.0, 1.0)
    pastel = (pastel_f * 255).astype(np.uint8)

    # Optional sharpen to retain some detail when sharpness > 1.0
    if sharpness > 1.0:
        sharpened_kernel = _SHARP_KERNEL * (sharpness - 1.0) * 0.6 + np.eye(3, dtype=np.float32)
        try:
            pastel = cv2.filter2D(pastel, -1, sharpened_kernel)
        except Exception:
            # fallback: simple unsharp mask
            blurred = cv2.GaussianBlur(pastel, (3, 3), 0)
            pastel = cv2.addWeighted(pastel, 1.0 + 0.3 * (sharpness - 1.0), blurred, -0.3 * (sharpness - 1.0), 0)

    # Slight overall brightness/contrast tweak
    beta = int((brightness - 1.0) * 15)
    alpha = 1.0 + (contrast - 1.0) * 0.25
    pastel = cv2.convertScaleAbs(pastel, alpha=alpha, beta=beta)

    # Edge-aware blend: keep a subtle amount of original edges to avoid overly smeared result
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    mask = (edges_colored > 0).astype(np.uint8)
    # Blend: where mask==1 use a slightly sharpened original, else use pastel
    original_resized = small
    blended = pastel.copy()
    blended[mask[:, :, 0] == 1] = cv2.addWeighted(original_resized, 0.6, pastel, 0.4, 0)[mask[:, :, 0] == 1]

    # Resize back to original dimensions if downscaled
    if small.shape[:2] != image.shape[:2]:
        blended = cv2.resize(blended, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    return blended

# Enhance original photo with quality and color (fast processing)
def enhance_original(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, hue=0.0, noise_reduction=0.0):
    # Downscale for faster processing
    small = downscale_for_processing(image)
    
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue_shift = int(hue * 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=(brightness - 1.0) * 10)
    
    if sharpness > 1.0:
        sharpened_kernel = _SHARP_KERNEL * sharpness
        enhanced = cv2.filter2D(enhanced, -1, sharpened_kernel)
    
    if noise_reduction > 0:
        enhanced = cv2.medianBlur(enhanced, 3)
    
    # Resize back to original dimensions if downscaled
    if small.shape[:2] != image.shape[:2]:
        enhanced = cv2.resize(enhanced, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return enhanced

# Handle file selection and effect application
def open_file(effect, batch=False):
    global current_image, current_effect, cap, is_camera_open
    if is_camera_open:
        close_camera()
    if batch:
        directory = filedialog.askdirectory(title="Select Directory")
        if directory:
            images = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in images:
                file_path = os.path.join(directory, img_file)
                image = cv2.imread(file_path)
                current_image = image.copy()
                current_effect = effect.get()
                apply_effect()
    else:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            current_image = image.copy()
            current_effect = effect.get()
            apply_effect()

# Display result in the result area
def display_result(output):
    global save_button
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    else:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    output_pil = Image.fromarray(output)
    output_tk = ImageTk.PhotoImage(output_pil)
    result_label.config(image=output_tk)
    result_label.image = output_tk

    if 'save_button' in globals() and save_button.winfo_exists():
        save_button.pack_forget()
    
    def save_image():
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            output_pil.save(save_path)
            messagebox.showinfo("Saved", f"Image saved to {save_path}")

    save_button = tk.Button(root, text="Save Image", command=save_image, bg='#4CAF50', fg='white', font=("Helvetica", 10), activebackground='#45a049', bd=1, relief=tk.RAISED, padx=10, pady=5)
    save_button.pack(pady=10, side=tk.LEFT)

# Handle live camera feed with chosen effect
def open_camera(effect):
    global current_image, current_effect, cap, is_camera_open
    if is_camera_open:
        messagebox.showwarning("Camera Warning", "Camera is already open. Please close it first.")
        return

    for i in range(3):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                is_camera_open = True
                break
        except Exception:
            continue
    if not cap or not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open any camera. Please ensure a camera is connected.")
        return

    current_effect = effect.get()

    update_frame()  # Start the loop

def update_frame():
    global current_image
    if not is_camera_open:
        return

    ret, frame = cap.read()
    if not ret:
        return

    current_image = frame.copy()
    selected_effect = current_effect

    # Apply effect
    if selected_effect == "Pencil Sketch":
        output = pencil_sketch(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
    elif selected_effect == "Cartoonify":
        output = cartoonify_image(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
    elif selected_effect == "Pastel":
        output = pastel_image(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
    else:
        output = enhance_original(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())

    output = resize_image(output)
    display_result(output)

    # Schedule next frame at ~30 FPS
    root.after(33, update_frame)

def save_frame_from_camera():
    global current_image, current_effect
    if current_image is not None and is_camera_open:
        # Freeze frame for saving and history
        frozen_frame = current_image.copy()

        # Apply current effect
        if current_effect == "Pencil Sketch":
            output = pencil_sketch(frozen_frame, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
        elif current_effect == "Cartoonify":
            output = cartoonify_image(frozen_frame, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
        elif current_effect == "Pastel":
            output = pastel_image(frozen_frame, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
        else:
            output = enhance_original(frozen_frame, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())

        output = resize_image(output)

        # Save dialog
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            Image.fromarray(cv2.cvtColor(output, cv2.COLOR_RGB2BGR)).save(save_path)
            messagebox.showinfo("Saved", f"Image saved to {save_path}")
            add_to_history(output, current_effect, save_path)

def close_camera():
    global cap, is_camera_open, current_image
    if is_camera_open and cap is not None:
        cap.release()
        is_camera_open = False
        current_image = None
        result_label.config(image='', text="Show Results")
        if 'save_button' in globals() and save_button.winfo_exists():
            save_button.place_forget()
        if 'close_button' in globals() and close_button.winfo_exists():
            close_button.place_forget()

def add_to_history(output, effect, file_path):
    global history_items
    thumbnail_size = (100, 75)
    output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    output_pil.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(output_pil)
    history_items.append((output_pil, effect, file_path))

def show_history():
    history_window = Toplevel(root)
    history_window.title("History")
    history_window.geometry("800x600")
    history_window.configure(bg='#E0F7FA')

    canvas = tk.Canvas(history_window, bg='#E0F7FA', width=780, height=520)
    canvas.pack(pady=10, padx=10)

    scroll_y = tk.Scrollbar(history_window, orient="vertical", command=canvas.yview)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scroll_y.set)

    history_frame = tk.Frame(canvas, bg='#E0F7FA')
    canvas.create_window((0, 0), window=history_frame, anchor="nw")

    for i, (output_pil, effect, file_path) in enumerate(history_items):
        photo = ImageTk.PhotoImage(output_pil)
        btn = tk.Button(history_frame, image=photo, bg='#E0F7FA', bd=0, relief=tk.FLAT, padx=5, pady=5,
                        command=lambda p=output_pil, e=effect, f=file_path: save_from_history(p, e, f, history_window))
        btn.image = photo
        btn.grid(row=i // 4, column=i % 4, padx=10, pady=10)

    def update_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    history_frame.bind("<Configure>", update_scroll_region)

def save_from_history(output_pil, effect, file_path, window):
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")], initialfile=os.path.basename(file_path))
    if save_path:
        output_pil.save(save_path)
        messagebox.showinfo("Saved", f"Image saved to {save_path}")
        window.destroy()

# Apply effect with current slider values
def apply_effect():
    global current_image, current_effect
    if current_image is None or current_effect is None:
        return
    
    if current_effect == "Pencil Sketch":
        output = pencil_sketch(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
    elif current_effect == "Cartoonify":
        output = cartoonify_image(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
    elif current_effect == "Pastel":
        output = pastel_image(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
    else:  # Enhance Original
        output = enhance_original(current_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
    
    output = resize_image(output)
    display_result(output)

# Main GUI setup
def main():
    global root, result_label, save_button, close_button, brightness_val, contrast_val, saturation_val, quality_var, sharpness_val, hue_val, noise_reduction_val, history_items
    
    # Create and configure the main window
    root = tk.Tk()
    root.title("Advanced Image Converter")
    root.configure(bg='#F5F5F5')
    root.geometry("1200x800")

    # Set up a custom style for rounded corners
    style = ttk.Style()
    style.configure("Custom.TFrame", background='#E0F7FA', borderwidth=2, relief='flat')
    style.configure("Rounded.TButton", font=("Helvetica", 10), padding=6, relief='flat')

    # Canvas for background
    canvas = tk.Canvas(root, width=1200, height=800, bg='#E0F7FA', highlightthickness=0)
    canvas.pack()
    canvas.create_rectangle(10, 10, 1190, 790, fill='#B2EBF2', outline='', width=0)

    # Title
    title_label = Label(root, text="Advanced Image Converter", font=("Helvetica", 20, "bold"), bg='#B2EBF2', fg='#212121', pady=15, padx=20)
    title_label.place(relx=0.5, rely=0.05, anchor='center')

    # Control frame with rounded corners
    control_frame = ttk.Frame(root, style="Custom.TFrame")
    control_frame.place(relx=0.5, rely=0.17, anchor='center')
    canvas.create_rectangle(control_frame.winfo_x(), control_frame.winfo_y(), 
                           control_frame.winfo_x() + 500, control_frame.winfo_y() + 50, 
                           outline='#B2EBF2', width=2, dash=(4, 4))

    # Effect dropdown
    effect_var = tk.StringVar(value="Pencil Sketch")
    effect_label = Label(control_frame, text="Select Effect:", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121')
    effect_label.grid(row=0, column=0, padx=15, pady=10)
    effect_dropdown = ttk.Combobox(control_frame, textvariable=effect_var, values=["Pencil Sketch", "Cartoonify", "Pastel", "Enhance Original"], state="readonly", font=("Helvetica", 10), width=20)
    effect_dropdown.grid(row=0, column=1, padx=15, pady=10)

    # Method dropdown
    method_var = tk.StringVar(value="Camera")
    method_label = Label(control_frame, text="Select Method:", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121')
    method_label.grid(row=0, column=2, padx=15, pady=10)
    method_dropdown = ttk.Combobox(control_frame, textvariable=method_var, values=["Camera", "Upload Image", "Batch Process"], state="readonly", font=("Helvetica", 10), width=20)
    method_dropdown.grid(row=0, column=3, padx=15, pady=10)

    # Run button with rounded style
    run_button = ttk.Button(control_frame, text="Run", command=lambda: run_process(effect_var, method_var), style="Rounded.TButton", padding=6)
    run_button.grid(row=0, column=4, padx=15, pady=10)

    # Enhancement sliders in 3x2 grid
    slider_frame = ttk.Frame(root, style="Custom.TFrame")
    slider_frame.place(relx=0.5, rely=0.3, anchor='center')
    canvas.create_rectangle(250, 150, 250 + 1000, 150 + 150, outline='#B2EBF2', width=2, dash=(4, 4))


    brightness_val = tk.DoubleVar(value=1.0)
    contrast_val = tk.DoubleVar(value=1.0)
    saturation_val = tk.DoubleVar(value=1.0)
    sharpness_val = tk.DoubleVar(value=1.0)
    hue_val = tk.DoubleVar(value=0.0)
    noise_reduction_val = tk.DoubleVar(value=0.0)

    # Row 1: Brightness and Contrast
    Label(slider_frame, text="Brightness", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121').grid(row=0, column=0, padx=15, pady=5)
    Scale(slider_frame, from_=0.5, to=1.5, resolution=0.1, variable=brightness_val, orient=tk.HORIZONTAL, length=200, bg='#B2EBF2', troughcolor='#E0F7FA', highlightthickness=0, command=lambda x: schedule_apply_effect()).grid(row=0, column=1, padx=15, pady=5)
    Label(slider_frame, text="Contrast", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121').grid(row=0, column=2, padx=15, pady=5)
    Scale(slider_frame, from_=0.5, to=1.5, resolution=0.1, variable=contrast_val, orient=tk.HORIZONTAL, length=200, bg='#B2EBF2', troughcolor='#E0F7FA', highlightthickness=0, command=lambda x: schedule_apply_effect()).grid(row=0, column=3, padx=15, pady=5)

    # Row 2: Saturation and Sharpness
    Label(slider_frame, text="Saturation", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121').grid(row=1, column=0, padx=15, pady=5)
    Scale(slider_frame, from_=0.5, to=1.5, resolution=0.1, variable=saturation_val, orient=tk.HORIZONTAL, length=200, bg='#B2EBF2', troughcolor='#E0F7FA', highlightthickness=0, command=lambda x: schedule_apply_effect()).grid(row=1, column=1, padx=15, pady=5)
    Label(slider_frame, text="Sharpness", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121').grid(row=1, column=2, padx=15, pady=5)
    Scale(slider_frame, from_=1.0, to=2.0, resolution=0.1, variable=sharpness_val, orient=tk.HORIZONTAL, length=200, bg='#B2EBF2', troughcolor='#E0F7FA', highlightthickness=0, command=lambda x: schedule_apply_effect()).grid(row=1, column=3, padx=15, pady=5)

    # Row 3: Hue Shift and Noise Reduction
    Label(slider_frame, text="Hue Shift", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121').grid(row=2, column=0, padx=15, pady=5)
    Scale(slider_frame, from_=-0.5, to=0.5, resolution=0.1, variable=hue_val, orient=tk.HORIZONTAL, length=200, bg='#B2EBF2', troughcolor='#E0F7FA', highlightthickness=0, command=lambda x: schedule_apply_effect()).grid(row=2, column=1, padx=15, pady=5)
    Label(slider_frame, text="Noise Reduction", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121').grid(row=2, column=2, padx=15, pady=5)
    Scale(slider_frame, from_=0.0, to=1.0, resolution=0.1, variable=noise_reduction_val, orient=tk.HORIZONTAL, length=200, bg='#B2EBF2', troughcolor='#E0F7FA', highlightthickness=0, command=lambda x: schedule_apply_effect()).grid(row=2, column=3, padx=15, pady=5)

    # Quality display with rounded corners
    quality_frame = ttk.Frame(root, style="Custom.TFrame")
    quality_frame.place(relx=0.5, rely=0.75, anchor='center')
    canvas.create_rectangle(quality_frame.winfo_x(), quality_frame.winfo_y(), 
                           quality_frame.winfo_x() + 200, quality_frame.winfo_y() + 50, 
                           outline='#B2EBF2', width=2, dash=(4, 4))
    quality_var = tk.IntVar(value=90)
    quality_label = Label(quality_frame, text="Original Quality: 90%", font=("Helvetica", 12), bg='#E0F7FA', fg='#212121')
    quality_label.grid(row=0, column=0, padx=15, pady=10)

    # Result area with rounded corners
    result_frame = ttk.Frame(root, style="Custom.TFrame")
    result_frame.place(relx=0.5, rely=0.68, anchor='center')
    canvas.create_rectangle(result_frame.winfo_x(), result_frame.winfo_y(), 
                           result_frame.winfo_x() + 600, result_frame.winfo_y() + 400, 
                           outline='#B2EBF2', width=2, dash=(4, 4))
    canvas_result = tk.Canvas(result_frame, width=600, height=400, bg='#B2EBF2', highlightthickness=0)
    canvas_result.pack()
    canvas_result.create_rectangle(2, 2, 598, 398, outline='#212121', dash=(4, 4))
    result_label = Label(canvas_result, text="Show Results", font=("Helvetica", 14), bg='#B2EBF2', fg='#212121')
    result_label.place(relx=0.5, rely=0.5, anchor='center')

    # History button with rounded style
    history_button = ttk.Button(root, text="History", command=show_history, style="Rounded.TButton")
    history_button.place(relx=0.8, rely=0.9, anchor='center')

    # UI Buttons in bottom left corner with rounded style
    save_button = ttk.Button(root, text="Save Frame", command=save_frame_from_camera, style="Rounded.TButton", padding=6)
    save_button.place(relx=0.1, rely=0.8, anchor='center')

    close_button = ttk.Button(root, text="Close Current Process", command=close_camera, style="Rounded.TButton", padding=6)
    close_button.place(relx=0.1, rely=0.9, anchor='center')

    history_items = []

    def run_process(effect, method):
        if method.get() == "Camera":
            open_camera(effect)
        elif method.get() == "Batch Process":
            open_file(effect, batch=True)
        else:
            open_file(effect)

    root.mainloop()

if __name__ == "__main__":
    main()