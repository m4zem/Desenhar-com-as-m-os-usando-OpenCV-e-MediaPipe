import cv2
import mediapipe as mp
import numpy as np

# Inicializando o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Criando a tela de desenho
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espelhar a imagem
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Pegando as posições dos dedos
            index_finger_tip = hand_landmarks.landmark[8]  # Ponta do dedo indicador
            index_finger_base = hand_landmarks.landmark[5]  # Base do dedo indicador
            thumb_tip = hand_landmarks.landmark[4]  # Ponta do polegar
            middle_finger_tip = hand_landmarks.landmark[12]  # Ponta do dedo médio
            ring_finger_tip = hand_landmarks.landmark[16]  # Ponta do anelar
            pinky_tip = hand_landmarks.landmark[20]  # Ponta do mínimo
            
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Verificando se o dedo indicador está levantado:
            # O dedo indicador deve estar acima de sua própria base e acima dos outros dedos
            if (index_finger_tip.y < index_finger_base.y and
                index_finger_tip.y < middle_finger_tip.y and
                index_finger_tip.y < ring_finger_tip.y and
                index_finger_tip.y < pinky_tip.y and
                index_finger_tip.y < thumb_tip.y):
                
                cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)  # Desenha na tela

            # Desenhar a mão detectada na tela
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mesclar a tela de desenho com a câmera
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Desenho com o Dedo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
