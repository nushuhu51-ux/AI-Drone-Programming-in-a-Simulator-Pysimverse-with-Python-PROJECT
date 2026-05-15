import time
from pysimverse import Drone
import cv2
import numpy as np

# =====================================================
# DRONE SETUP
# =====================================================

drone = Drone()
drone.connect()

print("Connected to drone")

# Start stream
drone.streamon()
time.sleep(2)

print("Stream started")

# =====================================================
# CAMERA TEST
# =====================================================

while True:

    try:
        # Get frame
        frame = drone.get_frame()

        # Check if frame exists
        if frame is None:
            print("No frame received")
            continue

        # Print frame type once
        print("Frame Type:", type(frame))

        img = None

        # =============================================
        # CASE 1: NUMPY ARRAY
        # =============================================
        if isinstance(frame, np.ndarray):
            img = frame

        # =============================================
        # CASE 2: TUPLE
        # =============================================
        elif isinstance(frame, tuple):

            print("Tuple length:", len(frame))

            img = frame[0]

            # If first element is bytes
            if isinstance(img, (bytes, bytearray)):
                np_arr = np.frombuffer(
                    img,
                    np.uint8
                )

                img = cv2.imdecode(
                    np_arr,
                    cv2.IMREAD_COLOR
                )

        # =============================================
        # CASE 3: DICTIONARY
        # =============================================
        elif isinstance(frame, dict):

            print("Dictionary keys:",
                  frame.keys())

            if "frame" in frame:
                img = frame["frame"]

            elif "image" in frame:
                img = frame["image"]

            else:
                print(
                    "Unknown dict format"
                )
                continue

        # =============================================
        # CASE 4: BYTES
        # =============================================
        elif isinstance(
            frame,
            (bytes, bytearray)
        ):

            np_arr = np.frombuffer(
                frame,
                np.uint8
            )

            img = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

        else:
            print(
                "Unsupported type:",
                type(frame)
            )
            continue

        # =============================================
        # VALIDATE IMAGE
        # =============================================

        if img is None:
            print(
                "Failed to decode frame"
            )
            continue

        if not isinstance(
            img,
            np.ndarray
        ):
            print(
                "Image is not numpy array:",
                type(img)
            )
            continue

        # Resize image
        img = cv2.resize(
            img,
            (640, 480)
        )

        # Show camera
        cv2.imshow(
            "Drone Camera Test",
            img
        )

    except Exception as e:
        print("ERROR:", e)

    # Quit
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# =====================================================
# CLEANUP
# =====================================================

cv2.destroyAllWindows()