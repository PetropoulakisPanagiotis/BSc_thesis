from pymouse import PyMouse
import time

m = PyMouse()
while True:
    m.click(600, 424)
    #time.sleep(5)
    #m.click(790, 424)
    time.sleep(150)
