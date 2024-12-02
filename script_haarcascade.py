import cv2

car_cascade = cv2.CascadeClassifier('./cars.xml')

cap = cv2.VideoCapture('./../../../home/tiago/Área de Trabalho/video2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Carros detectados', frame)
    
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


""" import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap(image_path):
    img = cv2.imread(image_path)
    heatmap = np.zeros((img.shape[0], img.shape[1]))
    
    heatmap[100:300, 100:400] = 1  # Região quente
    heatmap[300:500, 100:400] = 0.6  # Região média
    heatmap[500:700, 100:400] = 0.3  # Região fria

    plt.imshow(img, alpha=0.6)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.savefig('heatmap_image.png', bbox_inches='tight', pad_inches=0)
    return 'heatmap_image.png'

st.title("Visão Computacional")
st.subheader("Heatmap do Ambiente")
uploaded_file = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagem enviada', use_column_width=True)

    heatmap_image = generate_heatmap(uploaded_file)
    heatmap = Image.open(heatmap_image)
    st.image(heatmap, caption='Heatmap gerado', use_column_width=True)

    st.write("### Tempo médio de permanência: **40 segundos dentro do ROI**")
    st.write("### Quantidade média de veículos por ciclo: **6 veículos dentro do ROI**")
    
    st.write("### Ciclo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Red_light_icon.svg/1200px-Red_light_icon.svg.png", width=50)
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Yellow_light_icon.svg/1200px-Yellow_light_icon.svg.png", width=50)
    with col3:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Green_light_icon.svg/1200px-Green_light_icon.svg.png", width=50)

    st.write("**44 segundos**") """