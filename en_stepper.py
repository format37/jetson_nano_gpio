import sys

sys.path.append('/opt/nvidia/jetson-gpio/lib/python')
sys.path.append('/opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO')

import Jetson.GPIO as GPIO
import time

def stepperTurnOff():
    GPIO.output(stepper_pin_a, GPIO.LOW)
    GPIO.output(stepper_pin_b, GPIO.LOW)
    GPIO.output(stepper_pin_c, GPIO.LOW)
    GPIO.output(stepper_pin_d, GPIO.LOW)
	
def stepperSet(stepperCurrentLevel):
    GPIO.output(stepper_pin_a, stepperLevels[stepperCurrentLevel][0])
    GPIO.output(stepper_pin_b, stepperLevels[stepperCurrentLevel][1])
    GPIO.output(stepper_pin_c, stepperLevels[stepperCurrentLevel][2])
    GPIO.output(stepper_pin_d, stepperLevels[stepperCurrentLevel][3])
	
encoder_state_a=GPIO.LOW
encoder_state_b=GPIO.LOW
encoder_position_current=0

def encoder_event_a(channel):
	global encoder_state_a
	encoder_state_a = GPIO.input(channel)
	global encoder_state_b
	global encoder_position_current
	if			(encoder_state_a==GPIO.HIGH and encoder_state_b==GPIO.LOW) or (encoder_state_a==GPIO.LOW and encoder_state_b==GPIO.HIGH):
		encoder_position_current+=1
	else:
		if	(encoder_state_a==GPIO.HIGH and encoder_state_b==GPIO.HIGH) or (encoder_state_a==GPIO.LOW and encoder_state_b==GPIO.LOW):
			encoder_position_current-=1

def encoder_event_b(channel):
	global encoder_state_b
	encoder_state_b = GPIO.input(channel)
	global encoder_state_a
	global encoder_position_current
	if			(encoder_state_a==GPIO.HIGH and encoder_state_b==GPIO.HIGH) or (encoder_state_a==GPIO.LOW and encoder_state_b==GPIO.LOW):
		encoder_position_current+=1
	else:
		if	(encoder_state_a==GPIO.LOW and encoder_state_b==GPIO.HIGH) or (encoder_state_a==GPIO.HIGH and encoder_state_b==GPIO.LOW):
			encoder_position_current-=1
	
stepper_pin_a = 11
stepper_pin_b = 12
stepper_pin_c = 15
stepper_pin_d = 16

GPIO.setmode(GPIO.BOARD)
GPIO.setup(stepper_pin_a, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(stepper_pin_b, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(stepper_pin_c, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(stepper_pin_d, GPIO.OUT, initial=GPIO.HIGH)

stepperTurnOff()
stepperCurrentLevel=0

encoder_pin_a=18
GPIO.setup(encoder_pin_a, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # button pin set as input
GPIO.add_event_detect(encoder_pin_a, GPIO.BOTH, callback=encoder_event_a)

encoder_pin_b=19
GPIO.setup(encoder_pin_b, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # button pin set as input
GPIO.add_event_detect(encoder_pin_b, GPIO.BOTH, callback=encoder_event_b)

hi=GPIO.HIGH
lo=GPIO.LOW
stepperLevels=[
  [hi,lo,lo,lo],
  [hi,lo,hi,lo],
  [lo,lo,hi,lo],
  [lo,hi,hi,lo],
  [lo,hi,lo,lo],
  [lo,hi,lo,hi],
  [lo,lo,lo,hi],
  [hi,lo,lo,hi]
];

#main
print("input command:")
print("0 - exit")
print("1 - left")
print("2 - right")
cmd=int(input())
while cmd!=0:
	for i in range(500):
		stepperSet(stepperCurrentLevel)
		if cmd==1:
			stepperCurrentLevel-=1
		else:
			stepperCurrentLevel+=1
		if stepperCurrentLevel>7:
			stepperCurrentLevel=0
		else:
			if stepperCurrentLevel<0:
				stepperCurrentLevel=7
		time.sleep(0.005)
	print(encoder_position_current)
	stepperTurnOff()
	cmd=int(input())
print("cleanup")
GPIO.cleanup()