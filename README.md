Projeto: Desenhar com o Dedo Usando OpenCV e MediaPipe 🎨👆
O que esse projeto faz?
Ele usa a câmera do computador para detectar sua mão em tempo real e, quando o dedo indicador estiver levantado, ele desenha na tela.

Tecnologias usadas:
Python – Linguagem de programação.
OpenCV – Biblioteca para processar imagens e capturar vídeo.
MediaPipe – Biblioteca do Google para detectar mãos.
NumPy – Para criar a tela onde desenhamos.
Passo 1: Configurar o ambiente
Antes de começar a programar, precisamos instalar as bibliotecas necessárias.

Instalar o Python 🐍

Se ainda não tem, baixe e instale o Python.
Durante a instalação, marque a opção "Add Python to PATH".
Instalar as bibliotecas 📦

Abra o terminal (Prompt de Comando ou PowerShell) e rode os seguintes comandos:
bash
Copiar
Editar
pip install opencv-python numpy mediapipe
Isso instala as bibliotecas que vamos usar.
Passo 2: Capturar a imagem da câmera
O primeiro passo do código é capturar o vídeo da webcam.
No Python, usamos a biblioteca OpenCV (cv2) para isso.

python
Copiar
Editar
import cv2

cap = cv2.VideoCapture(0)  # Abre a câmera

while cap.isOpened():
    ret, frame = cap.read()  # Captura cada frame
    if not ret:
        break
    
    cv2.imshow("Camera", frame)  # Mostra a imagem da câmera

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Fecha ao apertar 'q'
        break

cap.release()
cv2.destroyAllWindows()
🔹 O que esse código faz?
✅ Abre a câmera.
✅ Captura cada frame do vídeo.
✅ Mostra a imagem da câmera em uma janela.
✅ Fecha a janela quando a tecla q é pressionada.

Passo 3: Detectar a mão usando MediaPipe
Agora, vamos adicionar o MediaPipe, que detecta mãos e identifica os dedos.

python
Copiar
Editar
import mediapipe as mp

mp_hands = mp.solutions.hands  # Inicializa o detector de mãos
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
🔹 O que esse código faz?
✅ Ativa o detector de mãos do MediaPipe.
✅ Define a precisão mínima para detectar a mão.

Agora, modificamos o loop da câmera para processar a imagem com MediaPipe:

python
Copiar
Editar
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte para RGB
result = hands.process(rgb_frame)  # Processa a imagem e detecta mãos

if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
🔹 O que esse código faz?
✅ Converte a imagem para RGB (porque o MediaPipe funciona melhor assim).
✅ Detecta as mãos na imagem.
✅ Desenha os pontos das mãos na tela.

Agora, quando você roda o código, ele mostra sua mão detectada na tela! 🖐️

Passo 4: Criar a tela de desenho
Vamos criar uma tela preta onde os desenhos serão feitos.

python
Copiar
Editar
import numpy as np
canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Cria uma tela preta
🔹 O que esse código faz?
✅ Cria uma imagem preta com resolução de 640x480 pixels.

Passo 5: Verificar se o dedo indicador está levantado
Cada dedo tem um índice numérico no MediaPipe:

Dedo	Índice no MediaPipe
Polegar	4
Indicador	8
Médio	12
Anelar	16
Mínimo	20
Podemos pegar a posição do dedo indicador e comparar se ele está levantado em relação aos outros.

python
Copiar
Editar
index_finger_tip = hand_landmarks.landmark[8]  # Ponta do indicador
index_finger_base = hand_landmarks.landmark[5]  # Base do indicador

# Verifica se o dedo indicador está levantado
if index_finger_tip.y < index_finger_base.y:
    x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)
    cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)  # Desenha um ponto na tela
🔹 O que esse código faz?
✅ Pega a posição do dedo indicador.
✅ Se a ponta estiver acima da base → o dedo está levantado!
✅ Desenha um ponto verde na tela.

Agora, o código só desenha quando o dedo indicador está levantado.

Passo 6: Mostrar o desenho na tela
Para exibir o desenho junto com a câmera, usamos:

python
Copiar
Editar
frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
🔹 O que esse código faz?
✅ Combina a câmera e a tela de desenho para mostrar o resultado final.

Passo 7: Código completo
Agora, juntamos tudo no código final:

python
Copiar
Editar
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_base = hand_landmarks.landmark[5]

            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if index_finger_tip.y < index_finger_base.y:
                cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    cv2.imshow("Desenho com o Dedo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Agora, quando você levantar o dedo indicador, ele desenha na tela! 🖌️✨

Próximos passos
✅ Testar e modificar o código para adicionar novas funcionalidades.
✅ Aprender sobre listas, loops e estruturas de decisão.
✅ Criar novas aplicações usando visão computacional!

Se precisar de mais explicações, só perguntar! 🚀🔥







