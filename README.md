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

### Phương pháp 1: Gaussian Blur + Sobel
- Gaussian Blur để làm mịn
- Sobel Edge Detection
- Phù hợp cho ảnh đơn giản, tốc độ nhanh

### Phương pháp 2: Bilateral Filter + Sobel
- Bilateral Filter
- Sobel Edge Detection
- Blending với ảnh gốc (30%)
- Tăng contrast
- Kết quả tự nhiên hơn, giống vẽ tay

### Phương pháp 3: Combined 1 + 2
- Tạo cả 2 phương pháp trên
- Blend 50-50 để kết hợp ưu điểm cả hai
- Nét vừa sắc (từ Gaussian) vừa mịn (từ Bilateral)
- Phù hợp cho ảnh phức tạp

### Phương pháp 4: Gaussian Blur + Laplacian
- Gaussian Blur để làm mịn
- Laplacian Edge Detection
- Tạo ra các đường biên mảnh và chi tiết hơn so với Sobel
- Phù hợp cho ảnh kiến trúc hoặc bản vẽ kỹ thuật

### Phương pháp 5: Bilateral Filter + Laplacian
- Bilateral Filter
- Laplacian Edge Detection
- Blending với ảnh gốc và tăng contrast
- Tạo ra bức tranh có chiều sâu và các mảng khối rõ ràng hơn

### Phương pháp 6: Combined 4 + 5
- Kết hợp kết quả của phương pháp 4 và 5
- Tối ưu hóa độ chi tiết, giảm thiểu nhiễu hạt tốt hơn bản Basic

## 🛠️ Công nghệ sử dụng

- Grayscale
- Gaussian Blur
- Sobel Edge Detection
- Laplacian Edge Detection
- Bilateral Filter
- Nearest Neighbor Resize

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
