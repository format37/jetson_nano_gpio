from environment import Encoder,Stepper,Environment
from debug import Debug
import tensorflow as tf
import numpy as np
import sys
import time
import math
from tensorboardX import SummaryWriter
import datetime

#Creating environment
stepper	= Stepper(11,12,15,16)
enc = Encoder(18,19,21)
env=Environment(stepper,enc)

step_current=0
stepper_states=np.zeros((1000,3))
for i in range(80):
	stepper.run(stepper.known_program[step_current])	
	step_current=step_current+1 if step_current<7 else 0	
	stepper_states[enc.position+500][0]=enc.position
	stepper_states[enc.position+500][1]=step_current
	report=str(enc.position)+" - "+str(step_current)
	print("\rc: {}".format(report), end="")

j=80	
for i in range(80):
	stepper.run(stepper.known_program[step_current])	
	step_current=step_current-1 if step_current>0 else 7
	stepper_states[enc.position+500][2]=step_current
	report=str(enc.position)+" - "+str(step_current)
	print("\rc: {}".format(report), end="")
	j=j-1

for i in range(190):
	print(stepper_states[i+500])
env.close()