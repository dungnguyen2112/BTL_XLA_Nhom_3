"""
Script test API - Chuyển ảnh thành tranh vẽ
"""
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

def test_convert(image_path: str, method: str = "advanced"):
    """Test endpoint /convert/"""
    print(f"\n{'='*60}")
    print(f"🧪 Test phương pháp: {method.upper()}")
    print(f"📁 Ảnh: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Không tìm thấy file: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'method': method,
            'blur_kernel': 5,
            'edge_threshold': 50,
            'max_size': 800
        }
        
        print("⏳ Đang xử lý...")
        start = time.time()
        
        try:
            response = requests.post(f"{API_URL}/convert/", files=files, data=data)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                output_file = f"test_output_{method}.png"
                with open(output_file, 'wb') as out:
                    out.write(response.content)
                print(f"✅ Thành công! Thời gian: {elapsed:.2f}s")
                print(f"💾 Đã lưu: {output_file}")
            else:
                print(f"❌ Lỗi {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Lỗi kết nối: {str(e)}")


def test_compare(image_path: str):
    """Test endpoint /compare/"""
    print(f"\n{'='*60}")
    print(f"🧪 Test so sánh 3 phương pháp")
    print(f"📁 Ảnh: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Không tìm thấy file: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'blur_kernel': 5,
            'edge_threshold': 50,
            'max_size': 800
        }
        
        print("⏳ Đang xử lý 3 phương pháp...")
        start = time.time()
        
        try:
            response = requests.post(f"{API_URL}/compare/", files=files, data=data)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                import base64
                from PIL import Image
                from io import BytesIO
                
                data = response.json()
                
                if data['success']:
                    print(f"✅ Thành công! Thời gian: {elapsed:.2f}s")
                    print(f"📊 Info: kernel={data['info']['blur_kernel']}, threshold={data['info']['edge_threshold']}")
                    
                    # Lưu cả 3 ảnh
                    for method, img_base64 in data['results'].items():
                        img_data = base64.b64decode(img_base64)
                        img = Image.open(BytesIO(img_data))
                        output_file = f"compare_{method}.png"
                        img.save(output_file)
                        print(f"💾 Đã lưu: {output_file}")
                else:
                    print(f"❌ Lỗi: {data}")
            else:
                print(f"❌ Lỗi {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")


def test_health():
    """Test endpoint /health"""
    print(f"\n{'='*60}")
    print("🧪 Test Health Check")
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server đang chạy: {data['message']}")
        else:
            print(f"❌ Lỗi {response.status_code}")
    except Exception as e:
        print(f"❌ Server không phản hồi: {str(e)}")
        print(f"💡 Hãy chạy: python app.py")


if __name__ == "__main__":
    print("🚀 Test FastAPI - Image to Sketch Converter")
    print(f"🌐 API URL: {API_URL}")
    
    # Test health check
    test_health()
    
    # Tìm ảnh test
    test_images = list(Path("image").glob("*.jpg")) + \
                  list(Path("image").glob("*.png")) + \
                  list(Path("image").glob("*.jpeg"))
    
    if not test_images:
        print("\n⚠️ Không tìm thấy ảnh trong thư mục 'image/'")
        print("💡 Hãy đặt ảnh test vào thư mục 'image/' và chạy lại")
        exit()
    
    # Lấy ảnh đầu tiên
    test_image = str(test_images[0])
    print(f"\n📷 Sử dụng ảnh test: {test_image}")
    
    # Test cả 3 phương pháp riêng lẻ
    test_convert(test_image, method="basic")
    test_convert(test_image, method="advanced")
    test_convert(test_image, method="combined")
    
    # Test so sánh 3 phương pháp cùng lúc
    test_compare(test_image)
    
    print(f"\n{'='*60}")
    print("✅ Hoàn thành tất cả test!")
    print(f"📂 Kết quả đã lưu:")
    print(f"   - test_output_basic.png")
    print(f"   - test_output_advanced.png")
    print(f"   - test_output_combined.png")
    print(f"   - compare_basic.png")
    print(f"   - compare_advanced.png")
    print(f"   - compare_combined.png")
