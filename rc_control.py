from pysimverse import Drone
import time

drone = Drone()
drone.connect()
drone.take_off()

left_right=50
forward_backward=0
up_down=0
yam=0

while True:
    drone.send_rc_control(left_right,
                          forward_backward,
                          up_down,
                          yam)



drone.land()
time.sleep(1)