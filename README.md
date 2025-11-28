# 🎨 Image to Sketch Converter - FastAPI Web App

API web để chuyển ảnh thành tranh vẽ

# Lớp INT13146-20251-01 Nhóm 03 - Bài Tập Lớn Xử Lý Ảnh

**Đề tài 4:** Xây dựng phần mềm chuyển ảnh thành tranh vẽ

**Thành viên nhóm:**
1. Chu Ngọc Thắng - B22DCCN807
2. Nguyễn Trí Dũng - B22DCCN135

## 🚀 Cài đặt và Chạy (Mở terminal ở dự án BTL_XLA)

### 1. Cài đặt dependencies

```bash
cd Code
pip install -r requirements.txt
```

### 2. Chạy server

```bash
python app.py
```

Hoặc dùng uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Truy cập

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### POST `/convert/`

Chuyển ảnh thành tranh vẽ

**Parameters:**
- `file` (file): File ảnh upload (JPG, PNG, BMP)
- `method` (string): `"basic"`, `"advanced"` hoặc `"combined"` (mặc định: `"advanced"`)
  - `basic`: Gaussian Blur + Sobel (nhanh)
  - `advanced`: Bilateral Filter + Sobel (chất lượng cao)
  - `combined`: Gộp cả 2 phương pháp (50-50)
- `blur_kernel` (int): Kích thước kernel làm mịn 3-15, lẻ (mặc định: 5)
- `edge_threshold` (float): Ngưỡng phát hiện biên 0-100 (mặc định: 50)
- `max_size` (int): Giới hạn kích thước ảnh, 0=không giới hạn (mặc định: 800)

**Response:** File ảnh PNG

**Ví dụ curl:**

```bash
# Phương pháp Advanced (khuyến nghị)
curl -X POST "http://localhost:8000/convert/" \
  -F "file=@test.jpg" \
  -F "method=advanced" \
  -F "blur_kernel=5" \
  -F "edge_threshold=50" \
  -F "max_size=800" \
  --output sketch_advanced.png

# Phương pháp Combined (gộp cả 2)
curl -X POST "http://localhost:8000/convert/" \
  -F "file=@test.jpg" \
  -F "method=combined" \
  -F "blur_kernel=5" \
  -F "edge_threshold=50" \
  -F "max_size=800" \
  --output sketch_combined.png
```

**Ví dụ Python:**

```python
import requests

with open('test.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'method': 'advanced',
        'blur_kernel': 5,
        'edge_threshold': 50,
        'max_size': 800
    }
    response = requests.post('http://localhost:8000/convert/', files=files, data=data)
    
    with open('sketch.png', 'wb') as out:
        out.write(response.content)
```

## 🎨 Phương pháp xử lý

### Phương pháp 1: Basic (Gaussian Blur + Sobel)
- Gaussian Blur để làm mịn
- Sobel Edge Detection
- Phù hợp cho ảnh đơn giản, tốc độ nhanh

### Phương pháp 2: Advanced (Bilateral Filter + Sobel)
- Bilateral Filter (edge-preserving)
- Sobel Edge Detection
- Blending với ảnh gốc (30%)
- Tăng contrast
- Kết quả tự nhiên hơn, giống vẽ tay

### Phương pháp 3: Combined (Gộp cả 2)
- Tạo cả 2 phương pháp trên
- Blend 50-50 để kết hợp ưu điểm cả hai
- Nét vừa sắc (từ Gaussian) vừa mịn (từ Bilateral)
- Phù hợp cho ảnh phức tạp
- **Lưu ý:** Chậm hơn gấp đôi vì xử lý 2 lần

## 🛠️ Thuật toán tự triển khai

**100% thuật toán tự viết, không dùng OpenCV/skimage:**
- Grayscale Conversion
- Gaussian Blur (separable - tối ưu)
- Sobel Edge Detection (separable - tối ưu)
- Bilateral Filter (Numba + Python fallback)
- Bilinear Resize
- Contrast Enhancement

## 📦 Cấu trúc project

```
BTL_XLA/
├── .gitignore
├── README.md
├── Báo cáo/
│   ├── Báo cáo BTL - XLA.pdf
    ├── Slide BTL - XLA.pdf
└── Code/
    ├── app.py                     # FastAPI web server
    ├── sketch_processor.py        # Module xử lý ảnh
    ├── requirements.txt           # Dependencies
    ├── deploy.txt                 # Tài liệu deploy
```

## 🌐 Deploy lên server

### Deploy với Docker

Tạo `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py sketch_processor.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build và chạy:

```bash
docker build -t sketch-converter .
docker run -p 8000:8000 sketch-converter
```

### Deploy lên Render.com (Free)

1. Push code lên GitHub
2. Tạo Web Service trên Render.com
3. Connect GitHub repo
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Deploy lên Railway.app (Free)

1. Push code lên GitHub
2. Tạo project trên Railway.app
3. Connect GitHub repo
4. Railway tự động detect và deploy

## 🔧 Tối ưu hiệu năng

1. **Giảm kích thước ảnh**: Set `max_size=800` để xử lý nhanh hơn
2. **Dùng Numba**: Cài `numba` để tăng tốc Bilateral Filter
3. **Chọn phương pháp Basic**: Nhanh hơn Advanced nhưng chất lượng thấp hơn
4. **Giảm blur_kernel**: Kernel nhỏ = xử lý nhanh hơn

## 🐛 Troubleshooting

**Lỗi: `ModuleNotFoundError: No module named 'fastapi'`**
```bash
pip install -r requirements.txt
```

**Lỗi: `Address already in use`**
```bash
# Đổi port
uvicorn app:app --port 8001
```

**Xử lý chậm:**
- Giảm `max_size` xuống 600-800
- Dùng phương pháp `basic`
- Cài `numba`: `pip install numba`

## 📝 License

Dự án học tập - INT13146 Xử lý ảnh
