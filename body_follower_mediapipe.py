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
model_path = "pose_landmarker.task"

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

detector = PoseLandmarker.create_from_options(options)

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

timestamp = 0

# -------------------- DRAW BODY --------------------
mp_pose = mp.solutions.pose

# -------------------- JUMP VARIABLES --------------------
previous_hip_y = None
jump_detected = False
JUMP_THRESHOLD = 0.05

# -------------------- MAIN LOOP --------------------
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

    h, w, _ = frame.shape

    # ---------------- BODY DETECTION ----------------
    if result.pose_landmarks:

        pose_landmarks = result.pose_landmarks[0]

        # ---------------- DRAW FULL BODY ----------------
        landmark_points = []

        for lm in pose_landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            landmark_points.append((x, y))

            # Draw landmark points
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        # Draw skeleton connections
        for connection in mp_pose.POSE_CONNECTIONS:

            start_idx = connection[0]
            end_idx = connection[1]

            x1, y1 = landmark_points[start_idx]
            x2, y2 = landmark_points[end_idx]

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

        # ---------------- FULL BODY CENTER ----------------
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]

        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]

        # Full body center
        center_x = (
            left_shoulder.x +
            right_shoulder.x +
            left_hip.x +
            right_hip.x
        ) / 4

        center_y = (
            left_shoulder.y +
            right_shoulder.y +
            left_hip.y +
            right_hip.y
        ) / 4

        # Convert to pixels
        cx = int(center_x * w)
        cy = int(center_y * h)

        # Draw center point
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        # ---------------- LEFT / RIGHT FOLLOW ----------------
        if center_x < 0.4:
            left_right = -SPEED

        elif center_x > 0.6:
            left_right = SPEED

        else:
            left_right = 0

        # ---------------- FORWARD / BACKWARD FOLLOW ----------------
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        hip_width = abs(right_hip.x - left_hip.x)

        body_width = (shoulder_width + hip_width) / 2

        # Person far
        if body_width < 0.15:
            forward_backward = SPEED

        # Person close
        elif body_width > 0.30:
            forward_backward = -SPEED

        else:
            forward_backward = 0

        # ---------------- JUMP DETECTION ----------------
        hip_center_y = (left_hip.y + right_hip.y) / 2

        if previous_hip_y is not None:

            movement = previous_hip_y - hip_center_y

            # Detect jump
            if movement > JUMP_THRESHOLD:

                jump_detected = True

                cv2.putText(
                    frame,
                    "JUMP DETECTED",
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                print("JUMP DETECTED")

                # Drone move up slightly
                up_down = 30

            else:
                jump_detected = False

        previous_hip_y = hip_center_y

        # ---------------- DISPLAY INFO ----------------
        cv2.putText(
            frame,
            "FULL BODY DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ---------------- SEND COMMAND ----------------
    drone.send_rc_control(
        left_right,
        forward_backward,
        up_down,
        yaw
    )

    # ---------------- DISPLAY CONTROLS ----------------
    cv2.putText(
        frame,
        f"LR: {left_right}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FB: {forward_backward}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"UD: {up_down}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    # ---------------- SHOW WINDOW ----------------
    cv2.imshow("Body Follower (MediaPipe)", frame)

    # ESC to quit
    key = cv2.waitKey(1)

    if key == 27:
        break

    time.sleep(0.03)

# ---------------- CLEANUP ----------------
drone.land()

cap.release()

cv2.destroyAllWindows()