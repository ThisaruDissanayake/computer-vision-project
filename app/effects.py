import cv2
import numpy as np

_PROCESSING_MAX_DIM = 800
_SHARP_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)


def downscale_for_processing(img, max_dim=_PROCESSING_MAX_DIM):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def pencil_sketch(image, brightness=1.0, contrast=1.0, saturation=1.0,
                  sharpness=1.0, hue=0.0, noise_reduction=0.0):
    """
    Pencil sketch effect with clean, smooth look using sigma=3 and tau=0.9
    """

   
    small = downscale_for_processing(image)
    if len(small.shape) == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = small

    # Smooth
    sigma = 2.0
    gray = cv2.GaussianBlur(gray, (0, 0), sigma)

    # brightness and contrast
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=(brightness - 1.0) * 100)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Create inverted image and apply dodge blend (τ = 0.9)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    tau = 0.9
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0 * tau)

  
    if sharpness > 1.0:
        sharpened_kernel = _SHARP_KERNEL * sharpness
        sketch = cv2.filter2D(sketch, -1, sharpened_kernel)

    
    if small.shape[:2] != image.shape[:2]:
        sketch = cv2.resize(sketch, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    return sketch



def cartoonify_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, hue=0.0, noise_reduction=0.0):
    small = downscale_for_processing(image)

    if noise_reduction > 0.2:
        smooth = cv2.bilateralFilter(small, d=5, sigmaColor=50, sigmaSpace=50)
    else:
        smooth = small.copy()

    data = smooth.reshape((-1, 3))
    data = np.float32(data)

    k = max(6, int(16 - noise_reduction * 8))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    quantized_data = centers[labels.flatten()]
    quantized = quantized_data.reshape(smooth.shape)

    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = int(hue * 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation * 1.5, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness * 1.1, 0, 255)
    cartoon_colors = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(gray_blur, 50, 150)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edges_mask = edges == 255
    cartoon_result = cartoon_colors.copy()
    cartoon_result[edges_mask] = [0, 0, 0]

    cartoon_result = cv2.convertScaleAbs(cartoon_result, alpha=contrast * 1.2, beta=(brightness - 1.0) * 15)

    if sharpness > 1.0:
        gaussian = cv2.GaussianBlur(cartoon_result, (0, 0), 2.0)
        cartoon_result = cv2.addWeighted(cartoon_result, 1.0 + (sharpness - 1.0) * 0.5, gaussian, -(sharpness - 1.0) * 0.5, 0)

    hsv_final = cv2.cvtColor(cartoon_result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_final[:, :, 1] = np.clip(hsv_final[:, :, 1] * 1.1, 0, 255)
    cartoon_final = cv2.cvtColor(hsv_final.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if small.shape[:2] != image.shape[:2]:
        cartoon_final = cv2.resize(cartoon_final, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    return cartoon_final
