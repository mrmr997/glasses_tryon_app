import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.title("🕶️ メガネ試着アプリ（mediapipeなし）")

# フォルダ内の透過PNGを読み込み
GLASSES_FOLDER = "glasses_images"

def load_glasses_images(folder):
    glasses_dict = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # アルファチャンネル付き
            if img is not None:
                glasses_dict[os.path.splitext(filename)[0]] = img
    return glasses_dict

glasses_images = load_glasses_images(GLASSES_FOLDER)
if not glasses_images:
    st.error("❌ 'glasses_images' フォルダにPNG画像が見つかりません。")
    st.stop()

# 透過合成関数
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

# サイドバーUI
uploaded_file = st.sidebar.file_uploader("📷 顔写真をアップロード", type=["jpg", "jpeg", "png"])
selected_glasses_name = st.sidebar.selectbox("🕶️ メガネを選択", list(glasses_images.keys()))
x_offset = st.sidebar.slider("▶️ 横位置調整", 0, 800, 100)
y_offset = st.sidebar.slider("🔽 縦位置調整", 0, 800, 100)
scale_factor = st.sidebar.slider("🔍 拡大率", 0.1, 5.0, 1.0, step=0.1)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    glasses_img = glasses_images[selected_glasses_name]

    # 位置と拡大率で単純に合成
    output = overlay_transparent(image_bgr.copy(), glasses_img, x_offset, y_offset, scale_factor)

    st.subheader("📸 アップロード画像")
    st.image(image, use_column_width=True)

    st.subheader("🕶️ メガネ試着結果")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)
else:
    st.info("顔写真をアップロードしてください。")
