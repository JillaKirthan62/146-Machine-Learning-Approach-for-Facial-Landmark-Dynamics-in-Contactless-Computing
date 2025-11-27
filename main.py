import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pyag
import time
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

pyag.FAILSAFE = False
pyag.PAUSE = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Thresholds
MOUTH_AR_THRESH = 0.63
MOUTH_FRAMES = 12
EYE_CLOSE = 0.18
SQUINT_THRESH = 0.20
WINK_FRAMES = 8
SQUINT_FRAMES = 10

# Counters & flags
MOUTH_COUNTER = 0
WINK_COUNTER = 0
SQUINT_COUNTER = 0
INPUT_MODE = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)

LEFT_EYE = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 386, 387, 263, 373, 380]
UPPER_LIP = [13, 14]
LOWER_LIP = [308, 78]
NOSE_TIP = 1

def eye_ratio(pts, idx):
    d1 = np.linalg.norm(pts[idx[1]] - pts[idx[5]])
    d2 = np.linalg.norm(pts[idx[2]] - pts[idx[4]])
    d3 = np.linalg.norm(pts[idx[0]] - pts[idx[3]])
    return (d1 + d2) / (2.0 * d3)

def mouth_ratio(pts):
    A = np.linalg.norm(pts[UPPER_LIP[0]] - pts[LOWER_LIP[0]])
    B = np.linalg.norm(pts[UPPER_LIP[1]] - pts[LOWER_LIP[1]])
    C = np.linalg.norm(pts[308] - pts[78])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_face_landmarks:
        lm = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark])
        leftEAR = eye_ratio(lm, LEFT_EYE)
        rightEAR = eye_ratio(lm, RIGHT_EYE)
        mar = mouth_ratio(lm)
        nose = tuple(lm[NOSE_TIP].astype(int))

        LEFT_CLOSED = leftEAR < EYE_CLOSE
        RIGHT_CLOSED = rightEAR < EYE_CLOSE

        # ========== WINK CLICK ==========
        if LEFT_CLOSED and not RIGHT_CLOSED:      # wink left
            WINK_COUNTER += 1
            if WINK_COUNTER > WINK_FRAMES:
                pyag.click(button="left")
                WINK_COUNTER = 0

        elif RIGHT_CLOSED and not LEFT_CLOSED:    # wink right
            WINK_COUNTER += 1
            if WINK_COUNTER > WINK_FRAMES:
                pyag.click(button="right")
                WINK_COUNTER = 0
        else:
            WINK_COUNTER = 0

        # ========== SQUINT (both eyes) → Scroll Mode ==========
        if leftEAR < SQUINT_THRESH and rightEAR < SQUINT_THRESH:
            SQUINT_COUNTER += 1
            if SQUINT_COUNTER > SQUINT_FRAMES:
                SCROLL_MODE = not SCROLL_MODE
                SQUINT_COUNTER = 0
        else:
            SQUINT_COUNTER = 0

        # ========== Mouth → Input Mode ==========
        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1
            if MOUTH_COUNTER > MOUTH_FRAMES:
                INPUT_MODE = not INPUT_MODE
                ANCHOR_POINT = nose
                MOUTH_COUNTER = 0
        else:
            MOUTH_COUNTER = 0

        # ========== Movement ==========
        if INPUT_MODE:
            x, y = ANCHOR_POINT
            nx, ny = nose
            drag = 16
            if SCROLL_MODE:
                if ny < y - 35: pyag.scroll(40)
                elif ny > y + 35: pyag.scroll(-40)
            else:
                if nx > x + 35: pyag.moveRel(drag, 0)
                elif nx < x - 35: pyag.moveRel(-drag, 0)
                if ny > y + 25: pyag.moveRel(0, drag)
                elif ny < y - 25: pyag.moveRel(0, -drag)

            cv2.putText(frame,
                        "SCROLL MODE" if SCROLL_MODE else "INPUT MODE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

    # FPS
    now = time.time()
    fps = 1 / (now - prev)
    prev = now
    cv2.putText(frame, f"FPS: {int(fps)}",
                (470, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Face Hands-Free Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
