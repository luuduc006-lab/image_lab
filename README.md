# ImageLab — Bài tập lớn Python

Ứng dụng web xử lý và phân tích ảnh sử dụng **Python, OpenCV, NumPy và Streamlit**. Dự án tập trung vào xử lý ảnh truyền thống, không huấn luyện mô hình AI.

## Tính năng

- Tải JPG/JPEG/PNG và so sánh ảnh gốc / kết quả.
- Ảnh xám, ảnh nhị phân, Gaussian Blur, Sharpen, Canny.
- Điều chỉnh độ sáng/tương phản, resize, xoay, lật.
- Histogram RGB, thông tin pixel, lưu PNG/JPEG.
- Giới hạn ảnh tải lên 10 MB, 20 megapixel.

## Yêu cầu

Python 3.11 hoặc 3.12, macOS/Windows/Linux.

## Cài đặt trên macOS

```bash
cd ImageLab
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Mở URL hiển thị trong Terminal (thường là http://localhost:8501). Dừng với `Ctrl+C`.

## Kiểm thử

```bash
python -m pytest -v
```

## Cấu trúc

```
ImageLab/
├── app.py
├── image_processing/
│   ├── __init__.py
│   ├── filters.py
│   ├── transformations.py
│   └── analysis.py
├── tests/test_processing.py
├── requirements.txt
├── requirements-dev.txt
├── .streamlit/config.toml
├── .gitignore
└── README.md
```

## Nguyên lý thuật toán

- **Grayscale:** chuyển RGB sang kênh cường độ sáng.
- **Binary Threshold:** nếu cường độ > ngưỡng → 255; ngược lại → 0.
- **Gaussian Blur:** lọc chập Gaussian để giảm nhiễu / làm mượt.
- **Sharpen:** sử dụng kernel 3x3 để nhấn mạnh khác biệt cục bộ.
- **Canny:** sử dụng các ngưỡng để phát hiện biên từ ảnh xám.
- **Brightness/Contrast:** `clip(pixel * contrast + brightness, 0, 255)`.
- **Histogram RGB:** đếm số pixel có từng giá trị 0...255 trên mỗi kênh.

## Giới hạn hiện tại

- Mỗi lần chỉ chọn **một phép xử lý** trên ảnh gốc; chưa có pipeline nhiều phép xử lý.
- Không có AI/machine learning, đăng nhập hay cơ sở dữ liệu.
- Xử lý ảnh trong bộ nhớ, không lưu ảnh lên máy chủ.

## Triển khai

1. Đưa repository lên GitHub.
2. Vào https://share.streamlit.io/ và tạo ứng dụng từ repository.
3. Chọn nhánh `main`, file `app.py`, triển khai.

## Gợi ý demo BTL

Tải ảnh mẫu → chọn từng thuật toán → thay đổi thông số → so sánh ảnh gốc/kết quả → histogram → tải xuống.

