
from pysimverse import Drone
import time

drone = Drone()
drone.connect()
drone.take_off()
drone.set_speed(50)
drone.move_forward(150)
#time.sleep(2)
#drone.move_backward(50)
#time.sleep(2)


drone.land()
time.sleep(1)