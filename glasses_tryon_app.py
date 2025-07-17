import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os


st.title("🕶️ メガネ試着アプリ")

mode = st.radio("モードを選んでください", ["📷 写真アップロード", "🎥 リアルタイム試着"])

if mode == "🎥 リアルタイム試着":
    st.warning("リアルタイム試着モードを選びました。以下を実行してください：")
    st.code("python glasses_tryon_realtime.py")
    st.stop()


# ====== 初期設定 ======
GLASSES_FOLDER = "glasses_images"  # 透過PNGを格納するフォルダ

# mediapipe 初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# メガネ画像読み込み
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
    st.error("❌ 'glasses_images' フォルダにPNG画像が見つかりませんでした。")
    st.stop()

# 合成関数
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

def try_on_glasses(image, glasses_img, x_offset=0, y_offset=0, scale_factor=1.5):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
        right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)

        eye_center_x = (left_x + right_x) // 2
        eye_center_y = (left_y + right_y) // 2
        glasses_width = right_x - left_x

        scale = glasses_width / glasses_img.shape[1] * scale_factor
        glasses_x = int(eye_center_x - (glasses_img.shape[1] * scale / 2)) + x_offset
        glasses_y = int(eye_center_y - (glasses_img.shape[0] * scale / 2.5)) + y_offset

        image = overlay_transparent(image, glasses_img, glasses_x, glasses_y, scale=scale)
    return image

# ====== UI レイアウト ======

# スタイリング
st.set_page_config(layout="wide", page_title="メガネ試着アプリ")
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    .stSlider > div[data-baseweb="slider"] > div {padding-top: 10px;}
    </style>
""", unsafe_allow_html=True)

# タイトル
st.title("🕶️ バーチャルメガネ試着アプリ")

# サイドバー設定
st.sidebar.header("🔧 調整パネル")

uploaded_file = st.sidebar.file_uploader("📷 顔写真をアップロード", type=["jpg", "jpeg", "png"])
selected_glasses_name = st.sidebar.selectbox("🕶️ メガネを選択", list(glasses_images.keys()))
x_offset = st.sidebar.slider("▶️ 横位置調整", -100, 100, 0)
y_offset = st.sidebar.slider("🔽 縦位置調整", -100, 100, 0)
scale_factor = st.sidebar.slider("🔍 拡大率", 0.5, 3.0, 1.5, step=0.1)

# 処理と表示
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    selected_img = glasses_images[selected_glasses_name]
    output_bgr = try_on_glasses(image_bgr.copy(), selected_img, x_offset, y_offset, scale_factor)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 アップロード画像")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("🕶️ 試着結果")
        st.image(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
