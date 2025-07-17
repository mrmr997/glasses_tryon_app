import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="🕶️ メガネ試着アプリ", layout="wide")
st.title("🕶️ バーチャルメガネ試着アプリ")

# ====== モード選択 ======
mode = st.radio("モードを選んでください", ["📷 写真アップロード", "🎥 リアルタイム試着"])
if mode == "🎥 リアルタイム試着":
    st.warning("リアルタイム試着は現在未対応です。写真アップロードでお試しください。")
    st.stop()

# ====== 初期設定 ======
GLASSES_FOLDER = "glasses_images"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def load_glasses_images(folder):
    glasses_dict = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                glasses_dict[os.path.splitext(filename)[0]] = img
    return glasses_dict

glasses_images = load_glasses_images(GLASSES_FOLDER)
if not glasses_images:
    st.error(f"❌ '{GLASSES_FOLDER}' フォルダにPNG画像が見つかりませんでした。")
    st.stop()

# ====== 合成関数 ======
def overlay_transparent(background, overlay, x, y, scale=1.0):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    roi = background[y:y+h, x:x+w]
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    roi[:] = (1.0 - mask) * roi + mask * overlay_img
    return background

def try_on_glasses_haar(image, glasses_img, x_offset=0, y_offset=0, scale_factor=1.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return image
    x, y, w, h = faces[0]
    glasses_width = int(w * scale_factor)
    scale = glasses_width / glasses_img.shape[1]
    glasses_x = x + x_offset
    glasses_y = y + int(h / 3.5) + y_offset
    return overlay_transparent(image, glasses_img, glasses_x, glasses_y, scale)

# ====== UI ======
st.sidebar.header("🔧 調整パネル")
uploaded_file = st.sidebar.file_uploader("📷 顔写真をアップロード", type=["jpg", "jpeg", "png"])
selected_glasses_name = st.sidebar.selectbox("🕶️ メガネを選択", list(glasses_images.keys()))
x_offset = st.sidebar.slider("▶️ 横位置調整", -1000, 1000, 10)
y_offset = st.sidebar.slider("🔽 縦位置調整", -100, 100, 0)
scale_factor = st.sidebar.slider("🔍 拡大率", 0.5, 3.0, 1.5, step=0.1)

# ====== 処理と表示 ======
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    selected_img = glasses_images[selected_glasses_name]
    output_bgr = try_on_glasses_haar(image_bgr.copy(), selected_img, x_offset, y_offset, scale_factor)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 アップロード画像")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("🕶️ 試着結果")
        st.image(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
