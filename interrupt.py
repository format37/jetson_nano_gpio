import sys
import time

sys.path.append('/opt/nvidia/jetson-gpio/lib/python')
sys.path.append('/opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO')

import Jetson.GPIO as GPIO

but_pin0 = 18
event_counter=0

def btn_event(channel):
    global event_counter
    event_counter=event_counter+1
	
# Pin Setup:
GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
GPIO.setup(but_pin0, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # button pin set as input
GPIO.add_event_detect(but_pin0, GPIO.BOTH, callback=btn_event)

for i in range(40):
    print(i," counter ",event_counter)
    time.sleep(1)
	
GPIO.cleanup()  # cleanup all GPIOs