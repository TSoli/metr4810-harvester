import time

from container import Container
from motors import CONTAINER_MOTOR_PWM_PIN, Servo

container_servo = Servo(CONTAINER_MOTOR_PWM_PIN)
c = Container(container_servo)
c.open()
time.sleep(2)
c.close()
