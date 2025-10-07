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

# Resize image or frame to fit the result area dynamically while maintaining aspect ratio
def resize_image(img):
    global result_label
    height, width = img.shape[:2]
    
    # Get result label dimensions dynamically
    try:
        result_label.update_idletasks()
        target_width = max(400, result_label.winfo_width() - 20)  # Leave some padding
        target_height = max(300, result_label.winfo_height() - 20)
    except:
        # Fallback to default size if there's an issue
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

# Enhanced cartoonify effect - sharp cartoon look like real animated cartoons
def cartoonify_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, hue=0.0, noise_reduction=0.0):
    # Downscale for faster processing
    small = downscale_for_processing(image)
    
    # Step 1: Light smoothing only if noise reduction is requested
    if noise_reduction > 0.2:
        smooth = cv2.bilateralFilter(small, d=5, sigmaColor=50, sigmaSpace=50)
    else:
        smooth = small.copy()
    
    # Step 2: K-means clustering for clean color regions (like real cartoons)
    # Reshape image to be a list of pixels
    data = smooth.reshape((-1, 3))
    data = np.float32(data)
    
    # Define criteria and apply K-means
    k = max(6, int(16 - noise_reduction * 8))  # 6-16 color clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape to original image shape
    centers = np.uint8(centers)
    quantized_data = centers[labels.flatten()]
    quantized = quantized_data.reshape(smooth.shape)
    
    # Step 3: Apply color adjustments in HSV for vibrant cartoon colors
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply hue shift
    hue_shift = int(hue * 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Boost saturation for cartoon vibrancy
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation * 1.5, 0, 255)
    
    # Adjust brightness
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness * 1.1, 0, 255)
    
    cartoon_colors = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Step 4: Create sharp, clean edges
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Create sharp edges without too much blur
    gray_blur = cv2.medianBlur(gray, 3)  # Minimal blur
    
    # Use Canny edge detection for cleaner lines
    edges = cv2.Canny(gray_blur, 50, 150)
    
    # Dilate edges to make them more prominent but not too thick
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert edges to 3-channel
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Step 5: Combine colors with edges
    # Invert edges so we can use them as a mask (white=keep color, black=draw edge)
    edges_mask = edges == 255
    
    cartoon_result = cartoon_colors.copy()
    # Draw black lines where edges are detected
    cartoon_result[edges_mask] = [0, 0, 0]
    
    # Step 6: Enhance contrast and brightness
    cartoon_result = cv2.convertScaleAbs(cartoon_result, alpha=contrast * 1.2, beta=(brightness - 1.0) * 15)
    
    # Step 7: Optional sharpening for crisp details
    if sharpness > 1.0:
        # Unsharp mask for crisp cartoon details
        gaussian = cv2.GaussianBlur(cartoon_result, (0, 0), 2.0)
        cartoon_result = cv2.addWeighted(cartoon_result, 1.0 + (sharpness - 1.0) * 0.5, gaussian, -(sharpness - 1.0) * 0.5, 0)
    
    # Step 8: Final saturation boost for cartoon pop
    hsv_final = cv2.cvtColor(cartoon_result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_final[:, :, 1] = np.clip(hsv_final[:, :, 1] * 1.1, 0, 255)
    cartoon_final = cv2.cvtColor(hsv_final.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Resize back to original dimensions if downscaled
    if small.shape[:2] != image.shape[:2]:
        cartoon_final = cv2.resize(cartoon_final, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return cartoon_final

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

# Global variables for batch processing
batch_images = []
batch_index = 0
batch_mode = False

# Handle file selection and effect application
def open_file(effect, batch=False):
    global current_image, current_effect, cap, is_camera_open, batch_images, batch_index, batch_mode
    if is_camera_open:
        close_camera()
    if batch:
        directory = filedialog.askdirectory(title="Select Directory")
        if directory:
            # Get all image files from directory
            image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                # Store batch information
                batch_images = [os.path.join(directory, f) for f in image_files]
                batch_index = 0
                batch_mode = True
                current_effect = effect.get()
                
                # Load and display first image
                load_batch_image(0)
                messagebox.showinfo("Batch Mode", f"Loaded {len(batch_images)} images. Use Next/Previous buttons to navigate.")
            else:
                messagebox.showwarning("No Images", "No image files found in the selected directory.")
    else:
        # Single file mode
        batch_mode = False
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            current_image = image.copy()
            current_effect = effect.get()
            apply_effect()

def load_batch_image(index):
    """Load and display a specific image from the batch"""
    global current_image, batch_images, batch_index
    if 0 <= index < len(batch_images):
        batch_index = index
        file_path = batch_images[index]
        image = cv2.imread(file_path)
        if image is not None:
            current_image = image.copy()
            apply_effect()
            # Update title to show current image info
            filename = os.path.basename(file_path)
            root.title(f"Advanced Image Converter - Batch Mode ({index + 1}/{len(batch_images)}) - {filename}")

def next_batch_image():
    """Navigate to next image in batch"""
    global batch_index, batch_images
    if batch_mode and batch_images:
        next_index = (batch_index + 1) % len(batch_images)  # Loop back to first
        load_batch_image(next_index)

def previous_batch_image():
    """Navigate to previous image in batch"""
    global batch_index, batch_images
    if batch_mode and batch_images:
        prev_index = (batch_index - 1) % len(batch_images)  # Loop to last
        load_batch_image(prev_index)

def save_all_batch_images():
    """Save all images in current batch with applied effects"""
    global batch_images, current_effect
    if not batch_images:
        messagebox.showwarning("No Images", "No batch images to save.")
        return
    
    # Ask user to select output directory
    output_dir = filedialog.askdirectory(title="Select Output Directory for Batch Save")
    if not output_dir:
        return
    
    saved_count = 0
    for i, image_path in enumerate(batch_images):
        try:
            # Load original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                continue
            
            # Apply current effect
            if current_effect == "Pencil Sketch":
                processed = pencil_sketch(original_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
            elif current_effect == "Cartoonify":
                processed = cartoonify_image(original_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
            elif current_effect == "Pastel":
                processed = pastel_image(original_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
            else:  # Enhance Original
                processed = enhance_original(original_image, brightness_val.get(), contrast_val.get(), saturation_val.get(), sharpness_val.get(), hue_val.get(), noise_reduction_val.get())
            
            # Generate output filename
            original_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{original_name}_{current_effect.replace(' ', '_').lower()}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Convert and save
            if len(processed.shape) == 2:  # Grayscale
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            Image.fromarray(processed_rgb).save(output_path)
            saved_count += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    messagebox.showinfo("Batch Save Complete", f"Successfully saved {saved_count}/{len(batch_images)} images to:\n{output_dir}")

def exit_batch_mode():
    """Exit batch processing mode"""
    global batch_mode, batch_images, batch_index, nav_frame
    batch_mode = False
    batch_images = []
    batch_index = 0
    root.title("Advanced Image Converter")
    # Clear the display
    result_label.config(image='', text="Show Results")
    # Hide navigation buttons
    if nav_frame and nav_frame.winfo_exists():
        nav_frame.place_forget()
    # Remove any existing buttons
    if 'image_save_button' in globals() and image_save_button and image_save_button.winfo_exists():
        image_save_button.destroy()
    if 'close_button' in globals() and close_button and close_button.winfo_exists():
        close_button.destroy()

# Display result in the result area
def display_result(output):
    global image_save_button, close_button
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    else:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    output_pil = Image.fromarray(output)
    output_tk = ImageTk.PhotoImage(output_pil)
    result_label.config(image=output_tk)
    result_label.image = output_tk

    # Remove existing buttons if they exist
    if 'image_save_button' in globals() and image_save_button and image_save_button.winfo_exists():
        image_save_button.destroy()
    
    def save_image():
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            output_pil.save(save_path)
            messagebox.showinfo("Saved", f"Image saved to {save_path}")
            # Add to history for uploaded images too
            if not is_camera_open:
                add_to_history(output, current_effect if current_effect else "Unknown", save_path)

    def close_current_process():
        global current_image, current_effect
        if batch_mode:
            exit_batch_mode()
        elif is_camera_open:
            close_camera()
        else:
            # For uploaded images, clear current image and reset display
            current_image = None
            current_effect = None
            result_label.config(image='', text="Show Results")
            # Remove buttons
            if image_save_button and image_save_button.winfo_exists():
                image_save_button.destroy()

    # Create both buttons for all modes (camera and uploaded images) in the button frame
    image_save_button = tk.Button(button_frame, text="Save Image", command=save_image, bg='#4CAF50', fg='white', font=("Helvetica", 10), activebackground='#45a049', bd=1, relief=tk.RAISED, padx=10, pady=5)
    image_save_button.grid(row=0, column=1, padx=10, pady=5)
    
    # Show close button for both camera and uploaded images in the button frame
    if not ('close_button' in globals() and close_button and close_button.winfo_exists()):
        close_button = tk.Button(button_frame, text="Close Current Process", command=close_current_process, bg='#f44336', fg='white', font=("Helvetica", 10), activebackground='#da190b', bd=1, relief=tk.RAISED, padx=10, pady=5)
        close_button.grid(row=0, column=0, padx=10, pady=5, sticky='w')
    
    # Show arrow navigation buttons if in batch mode
    if batch_mode and len(batch_images) > 1:
        # Create navigation frame if it doesn't exist
        if 'nav_frame' not in globals() or not nav_frame.winfo_exists():
            nav_frame = tk.Frame(root, bg='#B2EBF2', bd=2, relief='ridge')
        
        # Clear any existing navigation buttons
        for widget in nav_frame.winfo_children():
            widget.destroy()
        
        # Create left arrow button
        left_arrow = tk.Button(nav_frame, text="◀", command=previous_batch_image, bg='#2196F3', fg='white', font=("Arial", 16, "bold"), width=3, height=1, bd=1, relief='raised')
        left_arrow.pack(side=tk.LEFT, padx=8, pady=5)
        
        # Image counter in the middle
        image_counter = tk.Label(nav_frame, text=f"{batch_index + 1} / {len(batch_images)}", bg='#B2EBF2', font=("Helvetica", 11, "bold"), fg='#212121')
        image_counter.pack(side=tk.LEFT, padx=15, pady=5)
        
        # Create right arrow button
        right_arrow = tk.Button(nav_frame, text="▶", command=next_batch_image, bg='#2196F3', fg='white', font=("Arial", 16, "bold"), width=3, height=1, bd=1, relief='raised')
        right_arrow.pack(side=tk.LEFT, padx=8, pady=5)
        
        # Add Exit Batch Mode button
        exit_batch_btn = tk.Button(nav_frame, text="Exit Batch", command=exit_batch_mode, bg='#FF5722', fg='white', font=("Helvetica", 9, "bold"), width=8, height=1, bd=1, relief='raised')
        exit_batch_btn.pack(side=tk.LEFT, padx=8, pady=5)
        
        # Add Save All button
        save_all_btn = tk.Button(nav_frame, text="Save All", command=save_all_batch_images, bg='#4CAF50', fg='white', font=("Helvetica", 9, "bold"), width=8, height=1, bd=1, relief='raised')
        save_all_btn.pack(side=tk.LEFT, padx=8, pady=5)
        
        # Position navigation frame under the result area
        nav_frame.place(relx=0.5, rely=0.82, anchor='center')
    else:
        # Hide navigation frame if not in batch mode
        if 'nav_frame' in globals() and nav_frame.winfo_exists():
            nav_frame.place_forget()

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
    global cap, is_camera_open, current_image, image_save_button, close_button
    if is_camera_open and cap is not None:
        cap.release()
        is_camera_open = False
        current_image = None
        result_label.config(image='', text="Show Results")
        # Remove all buttons when closing camera
        if 'image_save_button' in globals() and image_save_button and image_save_button.winfo_exists():
            image_save_button.destroy()
        if 'close_button' in globals() and close_button and close_button.winfo_exists():
            close_button.destroy()

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
    global root, result_label, image_save_button, close_button, nav_frame, button_frame, brightness_val, contrast_val, saturation_val, quality_var, sharpness_val, hue_val, noise_reduction_val, history_items
    
    # Create and configure the main window
    root = tk.Tk()
    root.title("Advanced Image Converter")
    root.configure(bg='#F5F5F5')
    
    # Make window responsive
    root.geometry("1200x800")
    root.minsize(1000, 700)  # Minimum size
    root.maxsize(1920, 1080)  # Maximum size
    root.resizable(True, True)  # Allow resizing
    
    # Configure grid weights for responsiveness
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Set up a custom style for rounded corners
    style = ttk.Style()
    style.configure("Custom.TFrame", background='#E0F7FA', borderwidth=2, relief='flat')
    style.configure("Rounded.TButton", font=("Helvetica", 10), padding=6, relief='flat')

    # Create main responsive frame instead of fixed canvas
    main_frame = tk.Frame(root, bg='#E0F7FA')
    main_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
    
    # Configure main frame grid weights
    main_frame.grid_rowconfigure(0, weight=0)  # Title
    main_frame.grid_rowconfigure(1, weight=0)  # Controls
    main_frame.grid_rowconfigure(2, weight=0)  # Sliders
    main_frame.grid_rowconfigure(3, weight=1)  # Result area
    main_frame.grid_rowconfigure(4, weight=0)  # Buttons
    main_frame.grid_columnconfigure(0, weight=1)

    # Title
    title_label = Label(main_frame, text="Advanced Image Converter", font=("Helvetica", 20, "bold"), bg='#E0F7FA', fg='#212121', pady=15, padx=20)
    title_label.grid(row=0, column=0, pady=10, sticky='ew')

    # Control frame with rounded corners
    control_frame = ttk.Frame(main_frame, style="Custom.TFrame")
    control_frame.grid(row=1, column=0, pady=10, padx=20, sticky='ew')

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
    slider_frame = ttk.Frame(main_frame, style="Custom.TFrame")
    slider_frame.grid(row=2, column=0, pady=15, padx=20, sticky='ew')
    
    # Configure slider frame grid weights
    for i in range(4):
        slider_frame.grid_columnconfigure(i, weight=1)


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

    # Create a container for result area and quality info
    result_container = tk.Frame(main_frame, bg='#E0F7FA')
    result_container.grid(row=3, column=0, pady=15, padx=20, sticky='nsew')
    result_container.grid_rowconfigure(0, weight=1)
    result_container.grid_rowconfigure(1, weight=0)
    result_container.grid_columnconfigure(0, weight=1)

    # Result area with responsive sizing
    result_frame = tk.Frame(result_container, bg='#B2EBF2', bd=2, relief='ridge')
    result_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
    result_frame.grid_rowconfigure(0, weight=1)
    result_frame.grid_columnconfigure(0, weight=1)
    
    # Result label that will display images
    result_label = Label(result_frame, text="Show Results", font=("Helvetica", 14), bg='#B2EBF2', fg='#212121')
    result_label.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)



    # Button frame for responsive button layout
    button_frame = tk.Frame(main_frame, bg='#E0F7FA')
    button_frame.grid(row=4, column=0, pady=10, sticky='ew')
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)
    button_frame.grid_columnconfigure(2, weight=1)

    # History button
    history_button = ttk.Button(button_frame, text="History", command=show_history, style="Rounded.TButton")
    history_button.grid(row=0, column=2, padx=10, pady=5, sticky='e')

    # Note: Save and Close buttons are now created dynamically in display_result() function
    # This ensures they appear for both camera and uploaded image modes

    history_items = []
    image_save_button = None  # Initialize to avoid errors
    close_button = None  # Initialize to avoid errors
    nav_frame = None  # Initialize navigation frame

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