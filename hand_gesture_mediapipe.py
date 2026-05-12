import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pysimverse import Drone
import numpy as np
import time

# -------------------- DRONE --------------------
drone = Drone()
drone.connect()
drone.take_off()

SPEED = 40

# -------------------- MEDIAPIPE --------------------
model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

# -------------------- DRAW HAND --------------------
def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]

    # Draw lines
    for c in connections:
        x1 = int(hand_landmarks[c[0]].x * w)
        y1 = int(hand_landmarks[c[0]].y * h)

        x2 = int(hand_landmarks[c[1]].x * w)
        y2 = int(hand_landmarks[c[1]].y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # Draw points
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# -------------------- GESTURE DETECTION --------------------
def detect_gesture(hand_landmarks):

    lm = np.array([[p.x, p.y] for p in hand_landmarks])

    wrist = lm[0]
    middle_tip = lm[12]

    dist = np.linalg.norm(middle_tip - wrist)

    if dist > 0.25:
        return "OPEN"
    else:
        return "FIST"

# -------------------- MAIN LOOP --------------------
timestamp = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, timestamp)

    timestamp += 1

    # Default drone controls
    left_right = 0
    forward_backward = 0
    up_down = 0
    yaw = 0

    gesture = "NONE"

    h, w, _ = frame.shape

    # ---------------- HAND DETECTION ----------------
    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        # Draw hand
        draw_hand(frame, hand_landmarks)

        # Detect gesture
        gesture = detect_gesture(hand_landmarks)

        # ---------------- LEFT / RIGHT CONTROL ----------------
        hand_x = hand_landmarks[0].x

        # Move LEFT
        if hand_x < 0.4:
            left_right = -SPEED

        # Move RIGHT
        elif hand_x > 0.6:
            left_right = SPEED

        # Center = stop
        else:
            left_right = 0

        # ---------------- UP / DOWN CONTROL ----------------
        if gesture == "OPEN":
            up_down = SPEED

        elif gesture == "FIST":
            up_down = 0

        # Show center point
        cx = int(hand_x * w)
        cy = int(hand_landmarks[0].y * h)

        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

    # ---------------- SEND COMMAND ----------------
    drone.send_rc_control(
        left_right,
        forward_backward,
        up_down,
        yaw
    )

    # ---------------- DISPLAY ----------------
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"LR: {left_right}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow("Hand Gesture Drone Control", frame)

    # ESC to exit
    key = cv2.waitKey(1)

    if key == 27:
        break

    # Small delay for smoother control
    time.sleep(0.03)

# ---------------- CLEANUP ----------------
drone.land()

cap.release()

cv2.destroyAllWindows()