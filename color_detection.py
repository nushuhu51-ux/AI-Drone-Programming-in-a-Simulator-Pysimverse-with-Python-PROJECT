import time
from pysimverse import Drone
import cvzone
from cvzone.ColorModule import ColorFinder
import cv2
import numpy as np

# =====================================================
# DRONE SETUP
# =====================================================

drone = Drone()
drone.connect()

# Start stream
drone.streamon()
time.sleep(2)

# Take off
drone.take_off(takeoff_height=30)

# =====================================================
# COLOR DETECTION
# =====================================================

myColorFinder = ColorFinder(trackBar=False)

hsvVals = {'hmin': 0, 'smin': 51, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}


while True:

    try:
        # Get raw frame
        frame = drone.get_frame()

        if frame is None:
            print("No frame received")
            continue

        # If frame is bytes, decode it
        if isinstance(frame, bytes):
            np_arr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # If frame is tuple
        elif isinstance(frame, tuple):
            frame = frame[0]

            if isinstance(frame, bytes):
                np_arr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                img = frame

        else:
            img = frame

        # Check decoded image
        if img is None:
            print("Failed to decode frame.")
            continue

        # Resize image
        img = cv2.resize(img, (640, 480))

        # Detect color
        imgColor, mask = myColorFinder.update(img, hsvVals)

        # Stack images
        imgStack = cvzone.stackImages(
            [img, imgColor, mask],
            2,
            0.5
        )

        cv2.imshow("Drone Color Detection", imgStack)

    except Exception as e:
        print("Frame Error:", e)

    # Quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# =====================================================
# CLEANUP
# =====================================================

drone.land()
cv2.destroyAllWindows()