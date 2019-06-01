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
enc = Encoder(18,19,21,stepper)
env=Environment(stepper,enc)

debug=Debug(stepper,enc,env)
#program=[[1,0,0,0],[1,0,1,0],[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,1,0,1],[0,0,0,1],[1,0,0,1]]

def debug_out(enc):
	report=str(len(enc.cycles))+" - "+str(enc.position)+"    "
	print("\rc: {}".format(report), end="")

#for i in range(12000):
step_current=0
while len(enc.cycles)<3:
	stepper.run(stepper.known_program[step_current])
	#print(step_current,stepper.known_program[step_current])
	enc.stepper_states[enc.position]=step_current
	step_current=step_current+1 if step_current<7 else 0
	debug_out(enc)
	
	#debug.run_program(program)
	#debug.run()
	#stepper.run([1,0,0,0])
	#debug_out(enc)
	#stepper.run([1,0,1,0])
	#debug_out(enc)
	#stepper.run([0,0,1,0])
	#debug_out(enc)
	#stepper.run([0,1,1,0])
	#debug_out(enc)
	#stepper.run([0,1,0,0])
	#debug_out(enc)
	#stepper.run([0,1,0,1])
	#debug_out(enc)
	#stepper.run([0,0,0,1])
	#debug_out(enc)
	#stepper.run([1,0,0,1])
	#debug_out(enc)

print (enc.cycles)
#print (stepper.cycles)
for i in range(15):
	print(enc.stepper_states[i])
env.close()

#Building NN
n_inputs = 1
n_hidden1 = 32
n_outputs = 32

learning_rate = 0.0
discount_rate = 0.95

#for lr_step in range(1):

#learning_rate += 0.01
learning_rate = 0.1

writer = SummaryWriter("runs/"+datetime.datetime.now().strftime("%H-%M-%S_lr_")+str(learning_rate)+"_dr_"+str(discount_rate))

initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1=tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, kernel_initializer=initializer)
#logits = tf.layers.dense(X, n_outputs, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden1, n_outputs)
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=n_outputs)

y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
	gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
	gradient_placeholders.append(gradient_placeholder)
	grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
	discounted_rewards = np.zeros(len(rewards))
	cumulative_rewards = 0
	for step in reversed(range(len(rewards))):
		cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
		discounted_rewards[step] = cumulative_rewards
	return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
	all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()
	return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

def fourniture(four):
	niture=[four,four,four,four]
	return niture

#Training NN
n_games_per_update = 8
n_max_steps = 8
n_iterations = 1000
save_iterations = 10

position_last	= enc.position

with tf.Session() as sess:
	init.run()
	for iteration in range(n_iterations):
		#print("\rIteration: {}".format(iteration), end="")
		all_rewards = []
		all_gradients = []
		for game in range(n_games_per_update):
			current_rewards = []
			current_gradients = []
			#obs = env.reset()
			#obs = env.state()
			#obs = np.reshape(env.state(),(1,n_inputs))
			#obs = np.reshape(stepper.set,(1,n_inputs))
			obs=np.reshape(0,(1,n_inputs))
			for step in range(n_max_steps):				
				#if step==n_max_steps-1:
					#summary_str=mse_summary.eval(feed_dict={X: obs})
				tensorboard_step=iteration*n_games_per_update+game*n_max_steps+step
				#writer.add_scalar("action", action, tensorboard_step)
				#writer.add_scalar("gradients", gradients, tensorboard_step)
					#file_writer.add_summary(summary_str,tensorboard_step)
				action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs})				
				#obs, reward, done, info = env.step(action_val[0][0])
				#writer.add_summary(action_val, tensorboard_step)
				
				#action_set=np.reshape(np.asarray(action_val),4)
				action_set=np.reshape(np.asarray(action_val),32)
				#print(action_set)
				stepper_set=[0 if pin<50 else 1 for pin in action_set]				
				#stepper.run(stepper_set)
				#action_set=np.asarray(action_val)
				#stepper_set=[0 if pin<50 else 1 for pin in action_set]
				#stepper_set.reshape(8,4)
				stepper_program=np.reshape(stepper_set,(8,4))
				#print(stepper_program)
				#env.close()
				debug.run_program(stepper_program)
				#obs = env.state()
				#env_state=env.state()
				#env_state[0]=0
				#env_state[1]=0
				#obs = np.reshape(env_state,(1,n_inputs))
				#obs = np.reshape(stepper.set,(1,n_inputs))
				obs=np.reshape(0,(1,n_inputs))
				reward=(enc.position-position_last)
				#writer.add_scalar("X", X, tensorboard_step)
				
				writer.add_scalar("reward", reward, tensorboard_step)
				writer.add_scalar("enc_position", enc.position, tensorboard_step)
				#writer.add_scalars('stepper_set',{"a":stepper_set[0],"b":stepper_set[1],"c":stepper_set[2],"d":stepper_set[3]},tensorboard_step)
				#writer.add_scalars('stepper_program',{"a":stepper_program[0],"b":stepper_program[1],"c":stepper_program[2],"d":stepper_program[3]},tensorboard_step)
				writer.add_histogram('action_val', action_val, tensorboard_step) 
				writer.add_histogram('gradients_val_0', gradients_val[0], tensorboard_step) 
				position_last	= enc.position
				#print("iteration",iteration,"game",game,"step",step,"reward",reward)
				report=str(iteration)+" - "+str(game)+" - "+str(step)+" : "+str(enc.position)+" - "+str((reward if reward<0 else "+"+str(reward)))+"    "
				print("\rIteration: {}".format(report), end="")
				#print(env_state)
				
				current_rewards.append(reward)
				current_gradients.append(gradients_val)

				if int(enc.position)>1000:
					print("win")
					break

			all_rewards.append(current_rewards)
			all_gradients.append(current_gradients)

		all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
		feed_dict = {}
		for var_index, gradient_placeholder in enumerate(gradient_placeholders):
			mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
									  for game_index, rewards in enumerate(all_rewards)
										  for step, reward in enumerate(rewards)], axis=0)
			feed_dict[gradient_placeholder] = mean_gradients
		sess.run(training_op, feed_dict=feed_dict)
		if iteration % save_iterations == 0:
			saver.save(sess, "./my_policy_net_pg.ckpt")
writer.close()	
env.close()