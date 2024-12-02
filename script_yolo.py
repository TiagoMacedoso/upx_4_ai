""" Captura do ROI """

""" import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread("./../../../home/tiago/Área de Trabalho/img_roi.png")

# Lista para armazenar os pontos
roi_points = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Clique com o botão esquerdo
        roi_points.append((x, y))
        print(f"Ponto adicionado: ({x}, {y})")
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Marca o ponto na imagem
        cv2.imshow("Selecione o ROI", image)

# Mostrar a imagem e capturar os cliques
cv2.imshow("Selecione o ROI", image)
cv2.setMouseCallback("Selecione o ROI", select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exibir os pontos selecionados
print("Coordenadas do ROI:", roi_points) """







"""  VERSÃO RODANDO O VÍDEO VIA CV2 """


import cv2
import numpy as np
import os

# Diretório atual
path = os.getcwd()

# Carregar YOLOv3
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Nome das classes
classes = ["car"]

# Estrutura de rede
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = [(255, 0, 0)]  # Cor fixa para a classe "car"

# ROI Dataset 2 (Vídeo3)
roi_points = np.array([(554, 719), (669, 115), (780, 115), (1300, 720)])
#roi_points = np.array([(363, 650), (550, 240), (620, 240), (655, 658)])

# Caminho do vídeo
video_path = "./../../../home/tiago/Área de Trabalho/dataset.mp4"
cap = cv2.VideoCapture(video_path)

# Processar cada frame do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fim do vídeo

    height, width, channels = frame.shape

    # Criar uma máscara para o ROI
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    # Aplicar a máscara ao frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Pré-processamento
    blob = cv2.dnn.blobFromImage(masked_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Informações de objetos detectados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # Objeto encontrado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas da bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Verificar se a bounding box está dentro do ROI
                if cv2.pointPolygonTest(roi_points, (center_x, center_y), False) >= 0:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Exibição dos resultados
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    # Desenhar o ROI no frame
    frame_with_roi = cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Mostrar o frame processado
    cv2.imshow("Video", frame_with_roi)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()






""" VERSÃO STREAM LIT 1 """

""" import cv2
import numpy as np
import streamlit as st
import time
from collections import defaultdict

st.title("Visão Computacional - Análise de Veículos")
stframe = st.empty()
stats_placeholder = st.empty()

net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
classes = ["car"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = [(0, 255, 0)]

roi_points = np.array([(573, 721), (679, 123), (758, 122), (1143, 721)])

video_path = "./../../../home/tiago/Área de Trabalho/dataset.mp4"
cap = cv2.VideoCapture(video_path)

vehicle_tracker = {}
vehicle_count = 0
entry_times = defaultdict(float)
exit_times = defaultdict(float)
total_vehicles_in_roi = 0
vehicles_in_frame = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    blob = cv2.dnn.blobFromImage(masked_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    current_vehicles = set()
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if cv2.pointPolygonTest(roi_points, (center_x, center_y), False) >= 0:
                    boxes.append([x, y, w, h])
                    current_vehicles.add((center_x, center_y))

    for center in current_vehicles:
        if center not in vehicle_tracker:
            vehicle_count += 1
            vehicle_tracker[center] = vehicle_count
            entry_times[vehicle_count] = time.time()
        else:
            vehicles_in_frame.add(vehicle_tracker[center])

    for vehicle in list(vehicle_tracker.values()):
        if vehicle not in vehicles_in_frame and vehicle in entry_times:
            exit_times[vehicle] = time.time()

    for box, center in zip(boxes, current_vehicles):
        x, y, w, h = box
        label = f"Car {vehicle_tracker.get(center, '?')}"
        color = colors[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame_with_roi = cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

    stframe.image(frame_with_roi, channels="BGR", use_column_width=True)

    vehicles_inside = len(current_vehicles)
    total_vehicles_in_roi += vehicles_inside

    if exit_times:
        avg_time = np.mean([exit_times[v] - entry_times[v] for v in exit_times if exit_times[v] > entry_times[v]])
    else:
        avg_time = 0

    ### Estatísticas
    #- **Tempo médio de permanência no ROI:** {avg_time:.2f} segundos
    #- **Quantidade média de veículos por ciclo:** {vehicles_inside}
    #- **Total de veículos processados:** {vehicle_count}
    #"")

        REMOVER OS # acima

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() """






""" VERSÃO PARA GERAR O HEATMAP (AINDA COM PROBLEMAS) """

""" import cv2
import numpy as np
import os

# Diretório atual
path = os.getcwd()

# Carregar YOLOv3
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Nome das classes
classes = ["car"]

# Estrutura de rede
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = [(255, 0, 0)]  # Cor fixa para a classe "car"

# Coordenadas do ROI
roi_points = np.array([(573, 721), (679, 123), (758, 122), (1143, 721)])

# Caminho do vídeo
video_path = "./../../../home/tiago/Área de Trabalho/dataset.mp4"
cap = cv2.VideoCapture(video_path)

# Dimensões do vídeo
ret, frame = cap.read()
height, width, _ = frame.shape

# Matriz acumuladora para o heatmap
heatmap = np.zeros((height, width), dtype=np.float32)

# Processar cada frame do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fim do vídeo

    # Criar uma máscara para o ROI
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    # Aplicar a máscara ao frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Pré-processamento
    blob = cv2.dnn.blobFromImage(masked_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Informações de objetos detectados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # Objeto encontrado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas da bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Verificar se a bounding box está dentro do ROI
                if cv2.pointPolygonTest(roi_points, (center_x, center_y), False) >= 0:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Atualizar heatmap com a posição do centro do objeto
                    heatmap[center_y, center_x] += 1

    # Aplicar Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Exibição dos resultados
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    # Desenhar o ROI no frame
    frame_with_roi = cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Mostrar o frame processado
    cv2.imshow("Video", frame_with_roi)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Normalizar a matriz do heatmap
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = heatmap.astype(np.uint8)

# Aplicar colormap para o heatmap
heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Aplicar a máscara do ROI ao heatmap
masked_heatmap = cv2.bitwise_and(heatmap_colored, heatmap_colored, mask=mask)

# Mostrar e salvar o heatmap
cv2.imshow("Heatmap", masked_heatmap)
cv2.imwrite("heatmap.png", masked_heatmap)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
 """