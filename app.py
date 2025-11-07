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
from sketch_processor import (
    ImageProcessor, SketchEffectGenerator, maybe_downscale
)

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
                            <option value="basic">Phương pháp 1: Gaussian Blur + Sobel (Xám)</option>
                            <option value="advanced" selected>Phương pháp 2: Bilateral Filter + Sobel (Xám)</option>
                            <option value="combined">Phương pháp 3: Gộp cả 2 phương pháp (Xám)</option>
                            <option value="color">🎨 BONUS: Tranh vẽ có màu (Color Sketch)</option>
                        </select>
                    </div>

                    <div class="form-group" id="color-method-group" style="display:none;">
                        <label for="color_method">🖌️ Phương pháp tạo sketch xám (cho Color):</label>
                        <select id="color_method" name="color_method">
                            <option value="basic">Phương pháp 1: Gaussian Blur + Sobel</option>
                            <option value="advanced" selected>Phương pháp 2: Bilateral Filter + Sobel</option>
                            <option value="combined">Phương pháp 3: Gộp cả 2 phương pháp</option>
                        </select>
                    </div>

                    <div class="form-group" id="color-intensity-group" style="display:none;">
                        <label for="color_intensity">🎨 Độ đậm màu (0.0-1.0):</label>
                        <input type="number" id="color_intensity" name="color_intensity" value="0.6" min="0" max="1" step="0.1">
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
            // Show/hide color controls based on method
            document.getElementById('method').addEventListener('change', (e) => {
                const colorMethodGroup = document.getElementById('color-method-group');
                const colorIntensityGroup = document.getElementById('color-intensity-group');
                if (e.target.value === 'color') {
                    colorMethodGroup.style.display = 'block';
                    colorIntensityGroup.style.display = 'block';
                } else {
                    colorMethodGroup.style.display = 'none';
                    colorIntensityGroup.style.display = 'none';
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                const method = document.getElementById('method').value;
                const colorMethod = document.getElementById('color_method').value;
                const colorIntensity = document.getElementById('color_intensity').value;

                formData.append('file', fileInput.files[0]);
                
                // Xác định endpoint dựa trên method
                let endpoint = '/convert/';
                if (method === 'color') {
                    endpoint = '/convert_color/';
                    // Color sketch cần chọn phương pháp tạo sketch xám
                    formData.append('method', colorMethod);
                    formData.append('color_intensity', colorIntensity);
                } else {
                    formData.append('method', method);
                }

                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';

                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Lỗi xử lý ảnh');
                    }

                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);

                    // Hiển thị ảnh gốc
                    const originalUrl = URL.createObjectURL(fileInput.files[0]);
                    
                    // Tên phương pháp
                    let methodName = '';
                    if (method === 'color') {
                        const colorMethodNames = {
                            'basic': 'Phương pháp 1',
                            'advanced': 'Phương pháp 2',
                            'combined': 'Phương pháp 3'
                        };
                        methodName = `🎨 Color Sketch (${colorMethodNames[colorMethod]} - Tô màu)`;
                    } else {
                        const methodNames = {
                            'basic': 'Phương pháp 1 (Xám)',
                            'advanced': 'Phương pháp 2 (Xám)',
                            'combined': 'Phương pháp 3 (Xám)'
                        };
                        methodName = methodNames[method] || method;
                    }

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
    method: str = Form("advanced")
):
    """
    API endpoint chuyển ảnh thành tranh vẽ với thông số tối ưu cố định
    
    Parameters:
    - file: File ảnh upload
    - method: 'basic', 'advanced' hoặc 'combined'
    """
    try:
        # Thông số tối ưu cân bằng giữa chất lượng và tốc độ
        blur_kernel = 5
        edge_threshold = 30.0  # Giảm ngưỡng để giữ nhiều nét hơn
        max_size = 1200  # Tăng lên 1200 để giữ chi tiết tốt hơn
        
        # Đọc ảnh
        contents = await file.read()
        image = ImageProcessor.load_image_from_bytes(contents)
        
        # Downscale nếu cần
        if max_size > 0:
            image = maybe_downscale(image, max_side=max_size)
        
        # Xử lý ảnh với thông số tối ưu cho từng phương pháp
        if method == "basic":
            # Basic: edge_threshold thấp hơn để giữ nhiều nét
            sketch = SketchEffectGenerator.create_sketch_effect(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold * 0.8  # 30 * 0.8 = 24
            )
        elif method == "combined":
            # Phương pháp gộp: Tạo cả 2 và blend 50-50
            sketch_basic = SketchEffectGenerator.create_sketch_effect(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold * 0.8
            )
            
            sketch_advanced = SketchEffectGenerator.create_advanced_sketch(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold,
                blend_alpha=0.5,  # Tăng blend để giữ texture
                enhance_contrast=True
            )
            
            # Resize về cùng kích thước nếu khác nhau
            if sketch_basic.shape != sketch_advanced.shape:
                from sketch_processor import ImageResizer
                h_target, w_target = sketch_basic.shape
                sketch_advanced = ImageResizer.bilinear_resize(sketch_advanced, h_target, w_target)
            
            # Blend 50-50
            sketch = 0.5 * sketch_basic + 0.5 * sketch_advanced
            
        else:  # advanced
            # Advanced: blend_alpha cao hơn để giữ texture mịn
            sketch = SketchEffectGenerator.create_advanced_sketch(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold,
                blend_alpha=0.5,  # Tăng lên 0.5 để giữ chi tiết
                enhance_contrast=True
            )
        
        # Chuyển sang PIL Image
        pil_image = ImageProcessor.array_to_pil(sketch)
        
        # Trả về ảnh
        img_io = BytesIO()
        pil_image.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API đang hoạt động"}


@app.post("/compare/")
async def compare_methods(
    file: UploadFile = File(...)
):
    """
    So sánh cả 3 phương pháp cùng lúc với thông số tối ưu cố định
    Trả về JSON với 3 ảnh base64
    """
    try:
        import base64
        
        # Thông số tối ưu cân bằng giữa chất lượng và tốc độ
        blur_kernel = 5
        edge_threshold = 30.0
        max_size = 1200
        
        # Đọc ảnh
        contents = await file.read()
        image = ImageProcessor.load_image_from_bytes(contents)
        
        # Downscale nếu cần
        if max_size > 0:
            image = maybe_downscale(image, max_side=max_size)
        
        results = {}
        
        # Method 1: Basic - ngưỡng thấp để giữ nhiều nét
        sketch_basic = SketchEffectGenerator.create_sketch_effect(
            image, blur_kernel=blur_kernel, 
            edge_threshold=edge_threshold * 0.8
        )
        img_io = BytesIO()
        ImageProcessor.array_to_pil(sketch_basic).save(img_io, 'PNG')
        results['basic'] = base64.b64encode(img_io.getvalue()).decode()
        
        # Method 2: Advanced - blend cao để giữ texture
        sketch_advanced = SketchEffectGenerator.create_advanced_sketch(
            image, blur_kernel=blur_kernel, edge_threshold=edge_threshold,
            blend_alpha=0.5, enhance_contrast=True
        )
        img_io = BytesIO()
        ImageProcessor.array_to_pil(sketch_advanced).save(img_io, 'PNG')
        results['advanced'] = base64.b64encode(img_io.getvalue()).decode()
        
        # Method 3: Combined
        if sketch_basic.shape != sketch_advanced.shape:
            from sketch_processor import ImageResizer
            h_target, w_target = sketch_basic.shape
            sketch_advanced_resized = ImageResizer.bilinear_resize(sketch_advanced, h_target, w_target)
            sketch_combined = 0.5 * sketch_basic + 0.5 * sketch_advanced_resized
        else:
            sketch_combined = 0.5 * sketch_basic + 0.5 * sketch_advanced
        
        img_io = BytesIO()
        ImageProcessor.array_to_pil(sketch_combined).save(img_io, 'PNG')
        results['combined'] = base64.b64encode(img_io.getvalue()).decode()
        
        return {
            "success": True,
            "results": results,
            "info": {
                "blur_kernel": blur_kernel,
                "edge_threshold": edge_threshold,
                "image_shape": image.shape
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi so sánh: {str(e)}")


@app.post("/convert_color/")
async def convert_to_color_sketch(
    file: UploadFile = File(...),
    method: str = Form("advanced"),
    color_intensity: float = Form(0.6)
):
    """
    API endpoint tô màu lên tranh vẽ xám (Colorize Sketch) - Tính năng bonus
    
    Quy trình:
    1. Tạo sketch xám bằng phương pháp được chọn (basic/advanced/combined)
    2. Tô màu lên sketch xám dựa trên màu từ ảnh gốc
    
    Parameters:
    - file: File ảnh upload
    - method: 'basic', 'advanced' hoặc 'combined' (phương pháp tạo sketch xám)
    - color_intensity: Độ đậm của màu (0-1), 0=không màu, 1=màu đậm
    """
    try:
        from sketch_processor import ColorSketchGenerator
        
        # Thông số tối ưu cân bằng giữa chất lượng và tốc độ
        blur_kernel = 5
        edge_threshold = 30.0
        max_size = 1200
        
        # Đọc ảnh
        contents = await file.read()
        image = ImageProcessor.load_image_from_bytes(contents)
        original_image = image.copy()  # Giữ ảnh gốc để lấy màu
        
        # Downscale nếu cần
        if max_size > 0:
            image = maybe_downscale(image, max_side=max_size)
            original_image = maybe_downscale(original_image, max_side=max_size)
        
        # Validate color intensity
        color_intensity = max(0.0, min(1.0, color_intensity))
        
        # Bước 1: Tạo sketch xám theo phương pháp được chọn
        if method == "basic":
            sketch_gray = SketchEffectGenerator.create_sketch_effect(
                image,
                blur_kernel=blur_kernel,
                edge_threshold=edge_threshold * 0.8
            )
        elif method == "combined":
            sketch_basic = SketchEffectGenerator.create_sketch_effect(
                image, blur_kernel=blur_kernel, edge_threshold=edge_threshold * 0.8)
            
            sketch_advanced = SketchEffectGenerator.create_advanced_sketch(
                image, blur_kernel=blur_kernel, edge_threshold=edge_threshold,
                blend_alpha=0.5, enhance_contrast=True)
            
            if sketch_basic.shape != sketch_advanced.shape:
                from sketch_processor import ImageResizer
                h_target, w_target = sketch_basic.shape
                sketch_advanced = ImageResizer.bilinear_resize(sketch_advanced, h_target, w_target)
            
            sketch_gray = 0.5 * sketch_basic + 0.5 * sketch_advanced
        else:  # advanced
            sketch_gray = SketchEffectGenerator.create_advanced_sketch(
                image, blur_kernel=blur_kernel, edge_threshold=edge_threshold,
                blend_alpha=0.5, enhance_contrast=True)
        
        # Bước 2: Tô màu lên sketch xám
        color_sketch = ColorSketchGenerator.create_color_sketch(
            original_image,
            sketch_gray,
            color_intensity=color_intensity,
            blur_color=7
        )
        
        # Chuyển sang PIL Image
        pil_image = ImageProcessor.array_to_pil(color_sketch)
        
        # Trả về ảnh
        img_io = BytesIO()
        pil_image.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý color sketch: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
