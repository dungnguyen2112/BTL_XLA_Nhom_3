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
        """Resize ảnh bằng bilinear interpolation - tối ưu với vectorization"""
        is_color = len(image.shape) == 3
        if is_color:
            old_h, old_w, channels = image.shape
        else:
            old_h, old_w = image.shape
            channels = 1
            image = image[:, :, np.newaxis]
        
        # Tính toán scale factors
        scale_y = (old_h - 1) / (new_height - 1) if new_height > 1 else 0
        scale_x = (old_w - 1) / (new_width - 1) if new_width > 1 else 0
        
        # Vectorize: tính toán tất cả indices cùng lúc
        i_coords = np.arange(new_height, dtype=np.float32) * scale_y
        j_coords = np.arange(new_width, dtype=np.float32) * scale_x
        
        i0 = np.floor(i_coords).astype(np.int32)
        i1 = np.minimum(i0 + 1, old_h - 1)
        j0 = np.floor(j_coords).astype(np.int32)
        j1 = np.minimum(j0 + 1, old_w - 1)
        
        di = i_coords - i0
        dj = j_coords - j0
        
        # Broadcast để tính toán weights
        di_2d = di[:, np.newaxis]  # (new_h, 1)
        dj_2d = dj[np.newaxis, :]  # (1, new_w)
        
        w00 = (1 - di_2d) * (1 - dj_2d)  # (new_h, new_w)
        w01 = (1 - di_2d) * dj_2d
        w10 = di_2d * (1 - dj_2d)
        w11 = di_2d * dj_2d
        
        # Reshape để vectorize
        if is_color:
            output = np.zeros((new_height, new_width, channels), dtype=image.dtype)
        else:
            output = np.zeros((new_height, new_width), dtype=image.dtype)
        
        # Vectorized interpolation cho từng channel
        for c in range(channels):
            # Lấy các pixel values với advanced indexing
            p00 = image[i0[:, np.newaxis], j0[np.newaxis, :], c]
            p01 = image[i0[:, np.newaxis], j1[np.newaxis, :], c]
            p10 = image[i1[:, np.newaxis], j0[np.newaxis, :], c]
            p11 = image[i1[:, np.newaxis], j1[np.newaxis, :], c]
            
            # Bilinear interpolation
            val = w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11
            
            if is_color:
                output[:, :, c] = val
            else:
                output[:, :] = val
        
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

    @staticmethod
    def laplacian_edge_detection(image: np.ndarray) -> np.ndarray:
        """Phát hiện biên bằng toán tử Laplacian"""
        img = image.astype(np.float32, copy=False)
        # Kernel Laplacian
        laplacian = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)
        
        # Convolve với kernel Laplacian
        pad = 1
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
        h, w = img.shape
        output = np.zeros_like(img)
        
        for i in range(h):
            for j in range(w):
                region = padded[i:i+3, j:j+3]
                output[i, j] = np.sum(region * laplacian)
        
        return output


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
        h, w = image.shape
        total_pixels = h * w
        
        # Tự động giảm kernel size cho ảnh lớn để tăng tốc
        if total_pixels > 500000 and kernel_size > 3:  # > 500k pixels
            kernel_size = 3
        elif total_pixels > 300000 and kernel_size > 4:  # > 300k pixels
            kernel_size = 4
        
        # Thử Numba
        out_numba = EdgePreservingFilter._bilateral_filter_numba(
            image, kernel_size, sigma_spatial, sigma_intensity)
        if out_numba is not None:
            return out_numba.astype(image.dtype, copy=False)

        # Fallback Python
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

    @staticmethod
    def create_sketch_effect_laplacian(image: np.ndarray,
                                       blur_kernel: int = 5,
                                       edge_threshold: float = 50.0) -> np.ndarray:
        """Tạo hiệu ứng vẽ tay cơ bản với Laplacian"""
        gray = ImageProcessor.to_grayscale(image)
        blurred = ImageProcessor.gaussian_blur(gray, kernel_size=blur_kernel, sigma=1.0)

        laplacian_edges = EdgeDetector.laplacian_edge_detection(blurred)
        # Laplacian có thể có giá trị âm và dương, lấy giá trị tuyệt đối
        laplacian_abs = np.abs(laplacian_edges)
        laplacian_normalized = (laplacian_abs - laplacian_abs.min()) / (laplacian_abs.max() - laplacian_abs.min() + 1e-8) * 255
        edges = np.where(laplacian_normalized > edge_threshold, 255, 0)
        sketch = 255 - edges

        return sketch

    @staticmethod
    def create_advanced_sketch_laplacian(image: np.ndarray,
                                        blur_kernel: int = 5,
                                        edge_threshold: float = 50.0,
                                        blend_alpha: float = 0.3,
                                        enhance_contrast: bool = True) -> np.ndarray:
        """Tạo hiệu ứng vẽ tay nâng cao với bilateral filter và Laplacian"""
        gray = ImageProcessor.to_grayscale(image)
        blurred = EdgePreservingFilter.bilateral_filter(
            gray, kernel_size=blur_kernel, sigma_spatial=1.5, sigma_intensity=50.0)

        laplacian_edges = EdgeDetector.laplacian_edge_detection(blurred)
        laplacian_abs = np.abs(laplacian_edges)
        laplacian_normalized = (laplacian_abs - laplacian_abs.min()) / (laplacian_abs.max() - laplacian_abs.min() + 1e-8) * 255
        edges = np.where(laplacian_normalized > edge_threshold, 255, 0)
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

    @staticmethod
    def create_combined_sketch_laplacian(image: np.ndarray,
                                         blur_kernel: int = 5,
                                         edge_threshold: float = 50.0) -> np.ndarray:
        """Tạo hiệu ứng vẽ tay gộp cả 2 phương pháp với Laplacian"""
        # Phương pháp 1: Gaussian + Laplacian
        sketch_basic = SketchEffectGenerator.create_sketch_effect_laplacian(
            image, blur_kernel=blur_kernel, edge_threshold=edge_threshold * 0.8)
        
        # Phương pháp 2: Bilateral + Laplacian
        sketch_advanced = SketchEffectGenerator.create_advanced_sketch_laplacian(
            image, blur_kernel=blur_kernel, edge_threshold=edge_threshold,
            blend_alpha=0.5, enhance_contrast=True)
        
        # Resize về cùng kích thước nếu khác nhau
        if sketch_basic.shape != sketch_advanced.shape:
            h_target, w_target = sketch_basic.shape
            sketch_advanced = ImageResizer.bilinear_resize(sketch_advanced, h_target, w_target)
        
        # Blend 50-50
        sketch = 0.5 * sketch_basic + 0.5 * sketch_advanced
        
        return sketch



def maybe_downscale(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Giảm kích thước ảnh nếu quá lớn - tối ưu cho ảnh lớn"""
    if img is None:
        return img
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return img
    
    # Tính toán scale factor
    scale = max_dim / float(max_side)
    new_h = max(1, int(round(h / scale)))
    new_w = max(1, int(round(w / scale)))
    
    # Với ảnh rất lớn, có thể downscale nhiều lần để tăng tốc
    if max_dim > max_side * 2:
        # Downscale 2 lần để tăng tốc
        intermediate_size = int(max_side * 1.5)
        if max_dim > intermediate_size:
            scale1 = max_dim / float(intermediate_size)
            h1 = max(1, int(round(h / scale1)))
            w1 = max(1, int(round(w / scale1)))
            img = ImageResizer.bilinear_resize(img, h1, w1)
            h, w = h1, w1
    
    arr = ImageResizer.bilinear_resize(img, new_h, new_w)
    return arr
