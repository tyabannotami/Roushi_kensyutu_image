import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import os
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import read_image, visualize_object_predictions
from sahi.utils.file import download_from_url
from sahi.utils.yolov8 import download_yolov8s_model
import cv2

# YOLOv8の推論関数
def run_yolov8(image):
    #検出器の指定
    yolov8_model_path = "models/best.pt"
    # 結果を保存するフォルダのパス
    output_dir = 'kekka'  
    # SAHI用にモデルをラップ
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path=yolov8_model_path,
        confidence_threshold=0.75, #一致率がどの程度まで表示するか
        device='0'  # GPUを使用する場合
    )
    #検出
    result = get_prediction(image, model)

    # 検出結果の描画
    result.export_visuals(export_dir=output_dir, file_name="roushi")
    
    # 検出結果画像の読み込み
    result_image_path = os.path.join(output_dir, "roushi.png")
    result_image = cv2.imread(result_image_path)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
    
    return result_image

st.title('大熊老師探し')
#st.header('This is a header')
st.subheader('画像を選択してください。')
st.text('大熊老師が検出された場合、枠が表示されます。')

uploaded_file = st.file_uploader("ファイルを選択するかドロップしてください。",type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # 画像を読み込み
    image = Image.open(uploaded_file).convert("RGB")  # ここでRGB形式に変換
    #st.image(image, caption='アップロード画像', use_column_width=True)
    #st.write("")

    # YOLOv8を用いて検出を行う
    image_np = np.array(image)
    result_image = run_yolov8(image_np)

    # 検出結果の画像を表示
    st.image(result_image, caption='検出画像', use_column_width=True)