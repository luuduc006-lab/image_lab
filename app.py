"""ImageLab Streamlit app. Run: python -m streamlit run app.py"""
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from image_processing.analysis import image_details, rgb_histograms
from image_processing.filters import detect_edges, gaussian_blur, sharpen
from image_processing.transformations import (
    adjust_brightness_contrast, binary_threshold, flip_image,
    grayscale, resize_to_width, rotate_right_angle,
)

st.set_page_config(page_title="ImageLab | Python BTL", page_icon="🖼️", layout="wide")
st.title("🖼️ ImageLab – Xử lý và phân tích ảnh")
st.caption("Bài tập lớn Python | OpenCV · NumPy · Streamlit")

uploaded = st.file_uploader("Chọn ảnh JPG hoặc PNG (tối đa 10 MB)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Tải một hình ảnh lên để bắt đầu. Các thao tác không thay đổi tệp gốc.")
    st.stop()

if uploaded.size > 10 * 1024 * 1024:
    st.error("Ảnh vượt quá 10 MB. Hãy chọn ảnh nhỏ hơn.")
    st.stop()

try:
    original_pil = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    if original_pil.width * original_pil.height > 20_000_000:
        st.error("Ảnh quá lớn (hơn 20 triệu pixel). Hãy chọn ảnh nhỏ hơn.")
        st.stop()
    original = np.array(original_pil)
except (UnidentifiedImageError, OSError, ValueError):
    st.error("Không thể mở ảnh. Hãy chọn tệp JPG/PNG hợp lệ.")
    st.stop()

st.sidebar.header("Cài đặt xử lý")
operation = st.sidebar.selectbox("Chọn thuật toán", [
    "Ảnh gốc", "Chuyển xám", "Ảnh nhị phân", "Làm mờ Gaussian",
    "Làm sắc nét", "Phát hiện biên Canny", "Độ sáng / Tương phản",
    "Thay đổi kích thước", "Xoay ảnh", "Lật ảnh",
])

try:
    if operation == "Chuyển xám":
        result = grayscale(original)
    elif operation == "Ảnh nhị phân":
        threshold = st.sidebar.slider("Ngưỡng", 0, 255, 127)
        result = binary_threshold(original, threshold)
    elif operation == "Làm mờ Gaussian":
        kernel = st.sidebar.slider("Kích thước kernel", 1, 21, 5, step=2)
        result = gaussian_blur(original, kernel)
    elif operation == "Làm sắc nét":
        result = sharpen(original)
    elif operation == "Phát hiện biên Canny":
        low = st.sidebar.slider("Ngưỡng thấp", 0, 254, 100)
        high = st.sidebar.slider("Ngưỡng cao", low + 1, 255, max(low + 1, 200))
        result = detect_edges(original, low, high)
    elif operation == "Độ sáng / Tương phản":
        brightness = st.sidebar.slider("Độ sáng", -100, 100, 0)
        contrast = st.sidebar.slider("Tương phản", 0.1, 3.0, 1.0, 0.1)
        result = adjust_brightness_contrast(original, brightness, contrast)
    elif operation == "Thay đổi kích thước":
        width = st.sidebar.number_input("Chiều rộng mới (pixel)", min_value=1, max_value=5000,
                                        value=min(int(original.shape[1]), 5000))
        result = resize_to_width(original, int(width))
    elif operation == "Xoay ảnh":
        angle = st.sidebar.selectbox("Góc xoay theo chiều kim đồng hồ", [0, 90, 180, 270])
        result = rotate_right_angle(original, angle)
    elif operation == "Lật ảnh":
        direction = st.sidebar.selectbox("Chiều lật", ["None", "Horizontal", "Vertical"],
                                         format_func=lambda v: {"None": "Không lật", "Horizontal": "Ngang", "Vertical": "Dọc"}[v])
        result = flip_image(original, direction)
    else:
        result = original.copy()
except ValueError as exc:
    st.error(f"Thông số không hợp lệ: {exc}")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Ảnh gốc")
    st.image(original, width="stretch")
with right:
    st.subheader("Ảnh sau xử lý")
    st.image(result, width="stretch", clamp=True)

info_before, info_after = image_details(original), image_details(result)
st.write(f"**Ảnh gốc:** {info_before['width']} × {info_before['height']} px | "
         f"**Kết quả:** {info_after['width']} × {info_after['height']} px | "
         f"**Giá trị pixel trung bình:** {info_after['mean']:.1f}")

with st.expander("📊 Histogram RGB của ảnh gốc"):
    histograms = rgb_histograms(original)
    fig, ax = plt.subplots(figsize=(9, 3))
    for label, color in (("R", "red"), ("G", "green"), ("B", "blue")):
        ax.plot(histograms[label], color=color, label=label, linewidth=1)
    ax.set(xlabel="Giá trị pixel (0–255)", ylabel="Số pixel", xlim=(0, 255))
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

format_name = st.selectbox("Định dạng tải xuống", ["PNG", "JPEG"])
result_image = Image.fromarray(result)
buffer = BytesIO()
result_image.save(buffer, format=format_name, quality=95)
st.download_button("⬇️ Tải ảnh kết quả", data=buffer.getvalue(),
                   file_name=f"imagelab_result.{format_name.lower()}",
                   mime=f"image/{'jpeg' if format_name == 'JPEG' else 'png'}")
st.caption("Mỗi thao tác được áp dụng độc lập trên ảnh gốc. Không phải chuỗi nhiều bước.")
