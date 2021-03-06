import sys
sys.path.append('/opt/nvidia/jetson-gpio/lib/python')
sys.path.append('/opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO')
import Jetson.GPIO as GPIO
import numpy as np
import time

class Encoder:
	
	def __init__(self,phase_a_channel,phase_b_channel,phase_c_channel):
		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(phase_a_channel, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
		GPIO.setup(phase_b_channel, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
		GPIO.setup(phase_c_channel, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
		
		self.phase_a_state=GPIO.input(phase_a_channel)
		self.phase_b_state=GPIO.input(phase_b_channel)
		self.phase_c_state=GPIO.input(phase_b_channel)
		self.position=0
		#self.stepper_positions=np.zeros(1000)
		
		GPIO.add_event_detect(phase_a_channel, GPIO.BOTH, callback=self.phase_a_event)
		GPIO.add_event_detect(phase_b_channel, GPIO.BOTH, callback=self.phase_b_event)
		GPIO.add_event_detect(phase_c_channel, GPIO.BOTH, callback=self.phase_c_event)

	def phase_a_event(self,channel):
		self.phase_a_state = GPIO.input(channel)
		if (self.phase_a_state==GPIO.HIGH and self.phase_b_state==GPIO.LOW) or (self.phase_a_state==GPIO.LOW and self.phase_b_state==GPIO.HIGH):
			self.position-=1
		else:
			if (self.phase_a_state==GPIO.HIGH and self.phase_b_state==GPIO.HIGH) or (self.phase_a_state==GPIO.LOW and self.phase_b_state==GPIO.LOW):
				self.position+=1
				
	def phase_b_event(self,channel):
		self.phase_b_state = GPIO.input(channel)
		if (self.phase_a_state==GPIO.HIGH and self.phase_b_state==GPIO.HIGH) or (self.phase_a_state==GPIO.LOW and self.phase_b_state==GPIO.LOW):
			self.position-=1
		else:
			if (self.phase_a_state==GPIO.LOW and self.phase_b_state==GPIO.HIGH) or (self.phase_a_state==GPIO.HIGH and self.phase_b_state==GPIO.LOW):
				self.position+=1
				
	def phase_c_event(self,channel):
		self.phase_c_state = GPIO.input(channel)
		if self.position!=0 and self.phase_c_state==GPIO.HIGH:
			self.position=0
				
class Stepper:
	def __init__(self,channel_a,channel_b,channel_c,channel_d):
		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(channel_a, GPIO.OUT, initial=GPIO.HIGH)
		GPIO.setup(channel_b, GPIO.OUT, initial=GPIO.HIGH)
		GPIO.setup(channel_c, GPIO.OUT, initial=GPIO.HIGH)
		GPIO.setup(channel_d, GPIO.OUT, initial=GPIO.HIGH)
		self.channel_a	= channel_a
		self.channel_b	= channel_b
		self.channel_c	= channel_c
		self.channel_d	= channel_d
		self.known_program=[[1,0,0,0],[1,0,1,0],[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,1,0,1],[0,0,0,1],[1,0,0,1]]
		self.gpio_values = [[0,GPIO.LOW],[1,GPIO.HIGH]]
		self.disable()
		
	def disable(self):
		GPIO.output(self.channel_a, GPIO.LOW)
		GPIO.output(self.channel_b, GPIO.LOW)
		GPIO.output(self.channel_c, GPIO.LOW)
		GPIO.output(self.channel_d, GPIO.LOW)
		
	def run(self,set,sleep_time=0.005):
		GPIO.output(self.channel_a, self.gpio_values[set[0]][1])
		GPIO.output(self.channel_b, self.gpio_values[set[1]][1])
		GPIO.output(self.channel_c, self.gpio_values[set[2]][1])
		GPIO.output(self.channel_d, self.gpio_values[set[3]][1])
		self.set=set
		time.sleep(sleep_time)
		
	def run_program(self,set,sleep_time=0.005):
		for pos in range (8):
			GPIO.output(self.channel_a, self.gpio_values[set[pos][0]][1])
			GPIO.output(self.channel_b, self.gpio_values[set[pos][1]][1])
			GPIO.output(self.channel_c, self.gpio_values[set[pos][2]][1])
			GPIO.output(self.channel_d, self.gpio_values[set[pos][3]][1])
			time.sleep(sleep_time)
		
class Environment:
	def __init__(self,stepper,encoder):
		self.stepper=stepper
		self.encoder=encoder
		
	def close(self):
			GPIO.cleanup()
			print("normal exit")
			exit()