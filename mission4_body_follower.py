import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pysimverse import Drone
import numpy as np
import time
import urllib.request

# =========================================================
# SUPPRESS MEDIAPIPE WARNINGS
# =========================================================

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# =========================================================
# DOWNLOAD MODEL
# =========================================================

model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

model_path = "pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

# =========================================================
# DRONE SETUP
# =========================================================

drone = Drone()
drone.connect()
drone.take_off()

SPEED = 40

# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

detector = PoseLandmarker.create_from_options(options)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

timestamp = 0

# =========================================================
# MANUAL POSE CONNECTIONS
# =========================================================

POSE_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

# =========================================================
# VARIABLES
# =========================================================

previous_hip_y = None
previous_time = time.time()

hip_buffer = []
jump_frames = 0
cooldown = 0

JUMP_THRESHOLD = -3.5

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    left_right = 0
    forward_backward = 0
    up_down = 0
    yaw = 0

    h, w, _ = frame.shape

    if result.pose_landmarks:

        pose_landmarks = result.pose_landmarks[0]

        landmark_points = []

        for lm in pose_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmark_points.append((x, y))

            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        for c in POSE_CONNECTIONS:
            x1, y1 = landmark_points[c[0]]
            x2, y2 = landmark_points[c[1]]

            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # =================================================
        # BODY CENTER
        # =================================================

        ls = pose_landmarks[11]
        rs = pose_landmarks[12]
        lh = pose_landmarks[23]
        rh = pose_landmarks[24]

        center_x = (ls.x + rs.x + lh.x + rh.x) / 4

        # =================================================
        # LEFT / RIGHT CONTROL
        # =================================================

        if center_x < 0.4:
            left_right = -SPEED
        elif center_x > 0.6:
            left_right = SPEED

        # =================================================
        # FORWARD / BACKWARD CONTROL
        # =================================================

        body_width = (
            abs(rs.x - ls.x) + abs(rh.x - lh.x)
        ) / 2

        if body_width < 0.15:
            forward_backward = SPEED
        elif body_width > 0.30:
            forward_backward = -SPEED

        # =================================================
        # IMPROVED JUMP DETECTION
        # =================================================

        hip_center_y = (lh.y + rh.y) / 2
        current_time = time.time()
        delta_time = current_time - previous_time

        hip_buffer.append(hip_center_y)
        if len(hip_buffer) > 5:
            hip_buffer.pop(0)

        smoothed_hip_y = sum(hip_buffer) / len(hip_buffer)

        if previous_hip_y is not None and delta_time > 0:

            vertical_velocity = (
                smoothed_hip_y - previous_hip_y
            ) / delta_time

            cv2.putText(frame, f"Vy: {vertical_velocity:.2f}",
                        (20, 250), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), 2)

            # cooldown handling
            if cooldown > 0:
                cooldown -= 1

            if vertical_velocity < JUMP_THRESHOLD and cooldown == 0:
                jump_frames += 1
            else:
                jump_frames = 0

            if jump_frames >= 4:

                print("JUMP DETECTED")
                up_down = 30

                cv2.putText(frame, "JUMP DETECTED",
                            (20, 300), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                jump_frames = 0
                cooldown = 20

        previous_hip_y = smoothed_hip_y
        previous_time = current_time

    # =====================================================
    # SEND DRONE COMMANDS
    # =====================================================

    drone.send_rc_control(left_right, forward_backward, up_down, yaw)

    cv2.imshow("Drone Control + Stable Jump Detection", frame)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.03)

# =========================================================
# CLEANUP
# =========================================================

drone.land()
cap.release()
cv2.destroyAllWindows()