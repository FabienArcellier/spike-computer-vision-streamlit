import cv2
import mediapipe

cap = cv2.VideoCapture(0)
hands = mediapipe.solutions.hands.Hands()
landmark_layer = mediapipe.solutions.drawing_utils

while True:
    success, raw_img = cap.read()
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    output = hands.process(img)
    if output.multi_hand_landmarks:
        for i, hand_landmark in enumerate(output.multi_hand_landmarks):
            landmark_layer.draw_landmarks(img, hand_landmark, mediapipe.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
