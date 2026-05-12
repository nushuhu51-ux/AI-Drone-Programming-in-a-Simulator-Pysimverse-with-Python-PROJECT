from pysimverse import Drone
import time
import cv2

drone = Drone()
drone.connect()
time.sleep(5)
drone.streamon()
drone.take_off()
while True:
    frame, is_success = drone.get_frame()
    cv2.imshow("Drone Feed", frame)
    cv2.waitKey(5)


drone.land()
time.sleep(1)