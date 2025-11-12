"""
Sketch Processor Module - Xử lý ảnh thành tranh vẽ
Tách riêng các class xử lý để dùng trong FastAPI
"""
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import math

# Phát hiện Numba
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


class ImageProcessor:
    """Lớp xử lý ảnh với các thuật toán tự triển khai"""

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Đọc ảnh từ bytes"""
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        img_array = np.array(img, dtype=np.float32)
        return img_array

    @staticmethod
    def array_to_pil(image_array: np.ndarray) -> Image.Image:
        """Chuyển numpy array sang PIL Image"""
        img_normalized = np.clip(image_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_normalized)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Chuyển đổi ảnh RGB sang mức xám"""
        if len(image.shape) == 2:
            return image
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return gray

    @staticmethod
    def gaussian_1d_kernel(size: int, sigma: float) -> np.ndarray:
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        k = np.exp(-0.5 * (ax / sigma) ** 2)
        k = k / k.sum()
        return k.astype(np.float32)

    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Làm mịn ảnh bằng Gaussian blur (separable)"""
        if kernel_size < 3:
            return image
        k = ImageProcessor.gaussian_1d_kernel(kernel_size, sigma)
        pad = kernel_size // 2
        # Horizontal
        padded_h = np.pad(image, ((0, 0), (pad, pad)), mode='edge')
        horz = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), axis=1, arr=padded_h)
        # Vertical
        padded_v = np.pad(horz, ((pad, pad), (0, 0)), mode='edge')
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), axis=0, arr=padded_v)
        return out.astype(image.dtype)


class ImageResizer:
    """Lớp tự triển khai resize ảnh bằng nội suy song tuyến tính"""
    
    @staticmethod
    def bilinear_resize(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        """Resize ảnh bằng bilinear interpolation"""
        is_color = len(image.shape) == 3
        if is_color:
            old_h, old_w, channels = image.shape
            output = np.zeros((new_height, new_width, channels), dtype=image.dtype)
        else:
            old_h, old_w = image.shape
            channels = 1
            output = np.zeros((new_height, new_width), dtype=image.dtype)
            image = image[:, :, np.newaxis]
        
        scale_y = (old_h - 1) / (new_height - 1) if new_height > 1 else 0
        scale_x = (old_w - 1) / (new_width - 1) if new_width > 1 else 0
        
        for i in range(new_height):
            for j in range(new_width):
                old_i = i * scale_y
                old_j = j * scale_x
                
                i0 = int(np.floor(old_i))
                i1 = min(i0 + 1, old_h - 1)
                j0 = int(np.floor(old_j))
                j1 = min(j0 + 1, old_w - 1)
                
                di = old_i - i0
                dj = old_j - j0
                
                for c in range(channels):
                    p00 = image[i0, j0, c]
                    p01 = image[i0, j1, c]
                    p10 = image[i1, j0, c]
                    p11 = image[i1, j1, c]
                    
                    val = (1 - di) * (1 - dj) * p00 + \
                          (1 - di) * dj * p01 + \
                          di * (1 - dj) * p10 + \
                          di * dj * p11
                    
                    if is_color:
                        output[i, j, c] = val
                    else:
                        output[i, j] = val
        
        if not is_color:
            output = output.squeeze()
        
        return output


class EdgeDetector:
    """Lớp phát hiện biên"""

    @staticmethod
    def sobel_edge_detection(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phát hiện biên bằng Sobel (separable)"""
        img = image.astype(np.float32, copy=False)
        k_der = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        k_smt = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
        pad1 = 1

        # Gx
        padded_x = np.pad(img, ((0, 0), (pad1, pad1)), mode='edge')
        der_x = np.apply_along_axis(lambda m: np.convolve(m, k_der, mode='valid'), axis=1, arr=padded_x)
        padded_y = np.pad(der_x, ((pad1, pad1), (0, 0)), mode='edge')
        gx = np.apply_along_axis(lambda m: np.convolve(m, k_smt, mode='valid'), axis=0, arr=padded_y)

        # Gy
        padded_y = np.pad(img, ((pad1, pad1), (0, 0)), mode='edge')
        der_y = np.apply_along_axis(lambda m: np.convolve(m, k_der, mode='valid'), axis=0, arr=padded_y)
        padded_x = np.pad(der_y, ((0, 0), (pad1, pad1)), mode='edge')
        gy = np.apply_along_axis(lambda m: np.convolve(m, k_smt, mode='valid'), axis=1, arr=padded_x)

        magnitude = np.sqrt(gx * gx + gy * gy)
        return gx, gy, magnitude


class EdgePreservingFilter:
    """Lớp triển khai bilateral filter"""

    @staticmethod
    def _bilateral_filter_numba(image: np.ndarray, kernel_size: int, sigma_spatial: float, sigma_intensity: float):
        """Phiên bản Bilateral dùng Numba"""
        if not HAVE_NUMBA:
            return None
        
        pad = kernel_size // 2
        h, w = image.shape
        img32 = image.astype(np.float32, copy=False)

        ys = np.arange(-pad, pad + 1, dtype=np.float32)
        xs = np.arange(-pad, pad + 1, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        spatial = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma_spatial * sigma_spatial)).astype(np.float32)
        inv_2_sigma_i2 = np.float32(1.0 / (2.0 * sigma_intensity * sigma_intensity))
        padded = np.pad(img32, ((pad, pad), (pad, pad)), mode='edge')

        @nb.njit(cache=True, fastmath=True, nogil=True)
        def run(padded_arr, spatial_w, inv2sigI2, pad, h, w, ks):
            out = np.zeros((h, w), dtype=np.float32)
            for i in range(h):
                ip = i + pad
                for j in range(w):
                    jp = j + pad
                    cval = padded_arr[ip, jp]
                    num = 0.0
                    den = 0.0
                    for a in range(ks):
                        ia = ip + a - pad
                        for b in range(ks):
                            jb = jp + b - pad
                            v = padded_arr[ia, jb]
                            diff = v - cval
                            iw = math.exp(-(diff * diff) * inv2sigI2)
                            wgt = spatial_w[a, b] * iw
                            num += v * wgt
                            den += wgt
                    out[i, j] = cval if den <= 1e-12 else num / den
            return out

        try:
            out = run(padded, spatial, inv_2_sigma_i2, pad, h, w, kernel_size)
            return out
        except Exception:
            return None

    @staticmethod
    def bilateral_filter(image: np.ndarray, kernel_size: int = 5, 
                        sigma_spatial: float = 1.0, sigma_intensity: float = 50.0) -> np.ndarray:
        """Bilateral filter - làm mịn bảo toàn biên"""
        # Thử Numba
        out_numba = EdgePreservingFilter._bilateral_filter_numba(
            image, kernel_size, sigma_spatial, sigma_intensity)
        if out_numba is not None:
            return out_numba.astype(image.dtype, copy=False)

        # Fallback Python
        h, w = image.shape
        output = np.zeros_like(image)
        pad = kernel_size // 2
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')

        y_coords, x_coords = np.ogrid[-pad:pad+1, -pad:pad+1]
        spatial_weights = np.exp(-(x_coords**2 + y_coords**2) / (2.0 * sigma_spatial**2))
        inv_2_sigma_i2 = 1.0 / (2.0 * sigma_intensity * sigma_intensity)

        for i in range(h):
            for j in range(w):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                center_value = region[pad, pad]
                intensity_diff = region - center_value
                intensity_weights = np.exp(-(intensity_diff * intensity_diff) * inv_2_sigma_i2)
                weights = spatial_weights * intensity_weights
                denom = np.sum(weights)
                if denom <= 1e-12:
                    output[i, j] = center_value
                else:
                    output[i, j] = np.sum(region * weights) / denom

        return output


class SketchEffectGenerator:
    """Lớp tạo hiệu ứng vẽ tay"""

    @staticmethod
    def create_sketch_effect(image: np.ndarray,
                             blur_kernel: int = 5,
                             edge_threshold: float = 50.0) -> np.ndarray:
        """Tạo hiệu ứng vẽ tay cơ bản"""
        gray = ImageProcessor.to_grayscale(image)
        blurred = ImageProcessor.gaussian_blur(gray, kernel_size=blur_kernel, sigma=1.0)

        gx, gy, magnitude = EdgeDetector.sobel_edge_detection(blurred)
        magnitude_normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255
        edges = np.where(magnitude_normalized > edge_threshold, 255, 0)
        sketch = 255 - edges

        return sketch

    @staticmethod
    def create_advanced_sketch(image: np.ndarray,
                               blur_kernel: int = 5,
                               edge_threshold: float = 50.0,
                               blend_alpha: float = 0.3,
                               enhance_contrast: bool = True) -> np.ndarray:
        """Tạo hiệu ứng vẽ tay nâng cao với bilateral filter"""
        gray = ImageProcessor.to_grayscale(image)
        blurred = EdgePreservingFilter.bilateral_filter(
            gray, kernel_size=blur_kernel, sigma_spatial=1.5, sigma_intensity=50.0)

        gx, gy, magnitude = EdgeDetector.sobel_edge_detection(blurred)
        magnitude_normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255
        edges = np.where(magnitude_normalized > edge_threshold, 255, 0)
        sketch_edges = 255 - edges

        if blend_alpha <= 0.01:
            result = sketch_edges
        else:
            result = blend_alpha * sketch_edges + (1 - blend_alpha) * gray
            
            if enhance_contrast:
                result_min = result.min()
                result_max = result.max()
                if result_max - result_min > 1e-8:
                    result = (result - result_min) / (result_max - result_min) * 255
                    result = np.power(result / 255.0, 0.8) * 255

        return result



def maybe_downscale(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Giảm kích thước ảnh nếu quá lớn"""
    if img is None:
        return img
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return img
    scale = max_dim / float(max_side)
    new_h = int(round(h / scale))
    new_w = int(round(w / scale))
    arr = ImageResizer.bilinear_resize(img, new_h, new_w)
    return arr
