import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

finger_tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            finger_count = 0
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                if id in finger_tip_ids:
                    if handLms.landmark[id].y < handLms.landmark[id - 2].y:
                        finger_count += 1
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # determine hand type
            if hasattr(handLms, "handedness"):
                hand_handedness = handLms.handedness
                if hand_handedness.classification[0].label == 'Right':
                    hand_type = "Right Hand"
                else:
                    hand_type = "Left Hand"
                cv2.putText(img, hand_type, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.putText(img, f'Fingers: {finger_count}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
