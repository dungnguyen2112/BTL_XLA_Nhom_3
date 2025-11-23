"""
FastAPI Web App - Chuyển ảnh thành tranh vẽ
API endpoint cho dự án xử lý ảnh
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
import asyncio
import gc
from sketch_processor import (
    ImageProcessor, SketchEffectGenerator, maybe_downscale
)

# Global limits
MAX_SIDE = 800         # max long side in pixels to downscale to (giảm để tránh timeout)
MAX_UPLOAD_MB = 8      # max upload size in megabytes
MAX_PIXELS = 800 * 800  # max total pixels để tránh quá tải memory

app = FastAPI(
    title="Image to Sketch Converter",
    description="API chuyển ảnh thành tranh vẽ - Đề tài 4 INT13146",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Trang chủ với giao diện upload ảnh"""
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chuyển Ảnh Thành Tranh Vẽ</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 40px;
                font-size: 1.1em;
            }
            .upload-section {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            input[type="file"] {
                width: 100%;
                padding: 12px;
                border: 2px dashed #667eea;
                border-radius: 10px;
                background: white;
                cursor: pointer;
                transition: all 0.3s;
            }
            input[type="file"]:hover {
                border-color: #764ba2;
                background: #f0f0f0;
            }
            select, input[type="number"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                transition: border 0.3s;
            }
            select:focus, input[type="number"]:focus {
                outline: none;
                border-color: #667eea;
            }
            small {
                display: block;
                margin-top: 5px;
                color: #666;
                font-size: 12px;
                line-height: 1.4;
            }
            .btn-group {
                display: flex;
                gap: 15px;
                margin-top: 25px;
            }
            button {
                flex: 1;
                padding: 15px 30px;
                font-size: 18px;
                font-weight: 600;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s;
                color: white;
            }
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            .btn-secondary {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            .btn-secondary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(245, 87, 108, 0.3);
            }
            .results {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .result-card {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                text-align: center;
            }
            .result-card h3 {
                margin-bottom: 15px;
                color: #333;
            }
            .result-card img {
                width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            #loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #667eea;
                font-size: 18px;
                font-weight: 600;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .info-box h4 {
                color: #1976D2;
                margin-bottom: 5px;
            }
            .info-box p {
                color: #555;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 Chuyển Ảnh Thành Tranh Vẽ</h1>
            <p class="subtitle">Đề tài 4 - Xử lý ảnh INT13146</p>

            <div class="upload-section">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">📁 Chọn ảnh (JPG, PNG, BMP):</label>
                        <input type="file" id="file" name="file" accept="image/*" required>
                    </div>

                    <div class="form-group">
                        <label for="method">🎨 Phương pháp xử lý:</label>
                        <select id="method" name="method">
                            <optgroup label="Sobel Edge Detection">
                                <option value="basic">Phương pháp 1: Gaussian Blur + Sobel (Xám)</option>
                                <option value="advanced" selected>Phương pháp 2: Bilateral Filter + Sobel (Xám)</option>
                                <option value="combined">Phương pháp 3: Gộp cả 2 phương pháp Sobel (Xám)</option>
                            </optgroup>
                            <optgroup label="Laplacian Edge Detection">
                                <option value="laplacian_basic">Phương pháp 4: Gaussian Blur + Laplacian (Xám)</option>
                                <option value="laplacian_advanced">Phương pháp 5: Bilateral Filter + Laplacian (Xám)</option>
                                <option value="laplacian_combined">Phương pháp 6: Gộp cả 2 phương pháp Laplacian (Xám)</option>
                            </optgroup>
                        </select>
                    </div>

                    <div class="info-box">
                        <h4>⚙️ Tham số điều chỉnh (Tùy chọn)</h4>
                        <p>Bạn có thể điều chỉnh các tham số để tùy chỉnh kết quả. Để mặc định nếu không chắc chắn.</p>
                    </div>

                    <div class="form-group">
                        <label for="blur_kernel">🔲 Kích thước kernel làm mờ (3-15, mặc định: 5):</label>
                        <input type="number" id="blur_kernel" name="blur_kernel" min="3" max="15" step="1" value="5">
                        <small style="color:#666;font-size:12px;">Kernel lớn hơn = làm mờ nhiều hơn, xử lý chậm hơn</small>
                    </div>

                    <div class="form-group">
                        <label for="edge_threshold">📊 Ngưỡng phát hiện biên (10-100, mặc định: 30):</label>
                        <input type="number" id="edge_threshold" name="edge_threshold" min="10" max="100" step="1" value="30">
                        <small style="color:#666;font-size:12px;">Ngưỡng thấp = nhiều nét hơn, ngưỡng cao = ít nét hơn</small>
                    </div>

                    <div class="form-group" id="blend_alpha_group">
                        <label for="blend_alpha">🎭 Độ pha trộn (0.0-1.0, mặc định: 0.5):</label>
                        <input type="number" id="blend_alpha" name="blend_alpha" min="0.0" max="1.0" step="0.1" value="0.5">
                        <small style="color:#666;font-size:12px;">Chỉ áp dụng cho phương pháp Advanced/Combined. 0.0 = chỉ nét vẽ, 1.0 = nhiều texture gốc</small>
                    </div>

                    <div class="form-group">
                        <label for="max_size">📐 Giới hạn kích thước ảnh (400-2000, mặc định: 800):</label>
                        <input type="number" id="max_size" name="max_size" min="400" max="2000" step="100" value="800">
                        <small style="color:#666;font-size:12px;">Ảnh lớn hơn sẽ được thu nhỏ. Giá trị lớn = chất lượng cao nhưng xử lý chậm hơn</small>
                    </div>

                    <div class="btn-group">
                        <button type="submit" class="btn-primary">🚀 Xử lý ảnh</button>
                        <button type="button" class="btn-secondary" onclick="location.reload()">🔄 Làm mới</button>
                    </div>
                </form>
            </div>

            <div id="loading">
                <div class="spinner"></div>
                <p>Đang xử lý ảnh... Vui lòng đợi</p>
            </div>

            <div id="results" class="results"></div>
        </div>

        <script>
            // Ẩn/hiện blend_alpha dựa trên phương pháp được chọn
            function updateBlendAlphaVisibility() {
                const method = document.getElementById('method').value;
                const blendAlphaGroup = document.getElementById('blend_alpha_group');
                // Chỉ hiện cho advanced và combined methods
                if (method.includes('advanced') || method.includes('combined')) {
                    blendAlphaGroup.style.display = 'block';
                } else {
                    blendAlphaGroup.style.display = 'none';
                }
            }

            // Lắng nghe thay đổi phương pháp
            document.getElementById('method').addEventListener('change', updateBlendAlphaVisibility);
            // Khởi tạo lần đầu
            updateBlendAlphaVisibility();

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const formData = new FormData();
                const fileInput = document.getElementById('file');
                const method = document.getElementById('method').value;
                const blurKernel = document.getElementById('blur_kernel').value;
                const edgeThreshold = document.getElementById('edge_threshold').value;
                const blendAlpha = document.getElementById('blend_alpha').value;
                const maxSize = document.getElementById('max_size').value;

                formData.append('file', fileInput.files[0]);
                formData.append('method', method);
                formData.append('blur_kernel', blurKernel);
                formData.append('edge_threshold', edgeThreshold);
                formData.append('blend_alpha', blendAlpha);
                formData.append('max_size', maxSize);

                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';

                try {
                    const response = await fetch('/convert/', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Lỗi xử lý ảnh');
                    }

                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    const originalUrl = URL.createObjectURL(fileInput.files[0]);

                    const methodNames = {
                        'basic': 'Phương pháp 1: Sobel (Xám)',
                        'advanced': 'Phương pháp 2: Sobel (Xám)',
                        'combined': 'Phương pháp 3: Sobel (Xám)',
                        'laplacian_basic': 'Phương pháp 4: Laplacian (Xám)',
                        'laplacian_advanced': 'Phương pháp 5: Laplacian (Xám)',
                        'laplacian_combined': 'Phương pháp 6: Laplacian (Xám)'
                    };
                    const methodName = methodNames[method] || method;

                    document.getElementById('results').innerHTML = `
                        <div class="result-card">
                            <h3>📷 Ảnh gốc</h3>
                            <img src="${originalUrl}" alt="Original">
                        </div>
                        <div class="result-card">
                            <h3>🎨 Tranh vẽ</h3>
                            <p style="color:#666;margin-bottom:10px;font-size:14px;">${methodName}</p>
                            <img src="${imageUrl}" alt="Sketch">
                            <a href="${imageUrl}" download="sketch_${method}.png" style="display:inline-block;margin-top:15px;padding:10px 20px;background:#667eea;color:white;text-decoration:none;border-radius:5px;">💾 Tải xuống</a>
                        </div>
                    `;
                } catch (error) {
                    alert('Lỗi: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });

        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/convert/")
async def convert_to_sketch(
    file: UploadFile = File(...),
    method: str = Form("advanced"),
    blur_kernel: int = Form(5),
    edge_threshold: float = Form(30.0),
    blend_alpha: float = Form(0.5),
    max_size: int = Form(800)
):
    """
    API endpoint chuyển ảnh thành tranh vẽ với thông số có thể điều chỉnh
    
    Parameters:
    - file: File ảnh upload
    - method: 'basic', 'advanced', 'combined' (Sobel) hoặc 'laplacian_basic', 'laplacian_advanced', 'laplacian_combined' (Laplacian)
    - blur_kernel: Kích thước kernel làm mờ (3-15, mặc định: 5)
    - edge_threshold: Ngưỡng phát hiện biên (10-100, mặc định: 30.0)
    - blend_alpha: Độ pha trộn cho advanced/combined (0.0-1.0, mặc định: 0.5)
    - max_size: Giới hạn kích thước ảnh (400-2000, mặc định: 800)
    """
    try:
        # Validate các tham số đầu vào
        blur_kernel = max(3, min(15, int(blur_kernel)))  # Giới hạn 3-15
        edge_threshold = max(10.0, min(100.0, float(edge_threshold)))  # Giới hạn 10-100
        blend_alpha = max(0.0, min(1.0, float(blend_alpha)))  # Giới hạn 0.0-1.0
        max_size = max(400, min(2000, int(max_size)))  # Giới hạn 400-2000
        
        # Đọc ảnh (kiểm tra kích thước upload trước)
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File quá lớn. Giới hạn {MAX_UPLOAD_MB} MB.")

        image = ImageProcessor.load_image_from_bytes(contents)
        
        # Kiểm tra và downscale sớm để tránh timeout
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        total_pixels = h * w
        
        # Downscale nếu quá lớn (sử dụng max_size từ người dùng)
        if max_size > 0:
            image = maybe_downscale(image, max_side=max_size)
            if image.ndim == 2:
                h, w = image.shape
            else:
                h, w = image.shape[:2]
            total_pixels = h * w
        
        # Kiểm tra lại sau downscale (sử dụng max_size^2 làm giới hạn)
        max_pixels = max_size * max_size
        if total_pixels > max_pixels:
            # Downscale thêm nếu vẫn quá lớn
            scale = np.sqrt(max_pixels / total_pixels)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            from sketch_processor import ImageResizer
            image = ImageResizer.nearest_neighbor_resize(image, new_h, new_w)
        
        # Xử lý ảnh với thông số từ người dùng
        if method == "basic":
            # Basic Sobel: edge_threshold thấp hơn để giữ nhiều nét
            sketch = SketchEffectGenerator.create_sketch_effect(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold * 0.8
            )
        elif method == "combined":
            # Phương pháp gộp Sobel: Tạo cả 2 và blend với blend_alpha từ người dùng
            sketch_basic = SketchEffectGenerator.create_sketch_effect(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold * 0.8
            )
            
            sketch_advanced = SketchEffectGenerator.create_advanced_sketch(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold,
                blend_alpha=blend_alpha,
                enhance_contrast=True
            )
            
            # Resize về cùng kích thước nếu khác nhau
            if sketch_basic.shape != sketch_advanced.shape:
                from sketch_processor import ImageResizer
                h_target, w_target = sketch_basic.shape
                sketch_advanced = ImageResizer.nearest_neighbor_resize(sketch_advanced, h_target, w_target)
            
            # Blend với tỷ lệ từ người dùng
            sketch = blend_alpha * sketch_basic + (1 - blend_alpha) * sketch_advanced
            
        elif method == "laplacian_basic":
            # Basic Laplacian: edge_threshold thấp hơn để giữ nhiều nét
            sketch = SketchEffectGenerator.create_sketch_effect_laplacian(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold * 0.8
            )
        elif method == "laplacian_advanced":
            # Advanced Laplacian: sử dụng blend_alpha từ người dùng
            sketch = SketchEffectGenerator.create_advanced_sketch_laplacian(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold,
                blend_alpha=blend_alpha,
                enhance_contrast=True
            )
        elif method == "laplacian_combined":
            # Phương pháp gộp Laplacian: Tạo cả 2 và blend
            sketch = SketchEffectGenerator.create_combined_sketch_laplacian(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold
            )
        else:  # advanced (Sobel)
            # Advanced Sobel: sử dụng blend_alpha từ người dùng
            sketch = SketchEffectGenerator.create_advanced_sketch(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold,
                blend_alpha=blend_alpha,
                enhance_contrast=True
            )
        
        # Chuyển sang PIL Image
        pil_image = ImageProcessor.array_to_pil(sketch)
        
        # Cleanup memory
        del image, sketch
        gc.collect()
        
        # Trả về ảnh
        img_io = BytesIO()
        pil_image.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/png")
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Xử lý ảnh quá lâu. Vui lòng thử với ảnh nhỏ hơn.")
    except MemoryError:
        raise HTTPException(status_code=507, detail="Ảnh quá lớn, không đủ bộ nhớ. Vui lòng giảm kích thước ảnh.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API đang hoạt động"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)