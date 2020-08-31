import numpy as np 
import random
import time
import torch
from tqdm import tqdm

#######################################################################################

# Set parameters
l1 = 0.5
l2 = 0.5
bob1_radius = 0.1
bob2_radius = 0.1
target_radius = 0.05
ww = 600
wh = 600
linewidth = 2

speed = 0.05
actions = np.array([[-speed,-speed],
					[-speed,0],
					[-speed,speed],
					[0,-speed],
					[0,0],
					[0,speed],
					[speed,-speed],
					[speed,0],
					[speed,speed]])

#######################################################################################

def generate_random_angle():
	random_angle = np.random.random()*2*np.pi
	return random_angle

# Get the x and y coordinate of the end of the limb
def get_limb_end(x0, y0, length, angle):
	x1 = x0 + length*np.cos(angle)
	y1 = y0 + length*np.sin(angle)
	return x1, y1

def generate_target_position():
	target_angle = generate_random_angle()
	target_position_radius = np.random.random()*(l1+l2)
	target_x, target_y = get_limb_end(0,0,target_position_radius,target_angle)
	return np.array([target_x, target_y])

def generate_initial_state():
	theta1 = generate_random_angle()
	theta2 = generate_random_angle()
	#target_x, target_y = generate_target_position()
	target_x, target_y = 0.75, 0.75
	return np.array([theta1, theta2, target_x, target_y])

#######################################################################################

class RobotArmGame():
	def __init__(self):
		pass

	def reset(self):
		self.state = generate_initial_state()
		self.is_terminal = False
		return self.state

	# Move robot arm
	def move_arm(self, omega1, omega2):
		theta1 = self.state[0] + omega1
		theta2 = self.state[1] + omega2

		# Angles cannot exceed 2pi!
		if theta1 >= 2*np.pi:
			theta1 = theta1 - 2*np.pi
		if theta2 >= 2*np.pi:
			theta2 = theta2 - 2*np.pi

		# Angles cannot be less than 0
		if theta1 < 0:
			theta1 = theta1 + 2*np.pi
		if theta2 < 0:
			theta2 = theta2 + 2*np.pi

		self.state[0] = theta1
		self.state[1] = theta2

	def get_reward(self):
		target_x = self.state[2]
		target_y = self.state[3]
		theta1 = self.state[0]
		theta2 = self.state[1]

		x1 = l1*np.cos(theta1)
		x2 = l2*np.cos(theta2)
		y1 = l1*np.sin(theta1)
		y2 = l2*np.sin(theta2)

		bob_x = x1 + x2
		bob_y = y1 + y2

		distance_between_centers = np.sqrt((bob_x - target_x)**2 + (bob_y - target_y)**2)
		max_distance = bob2_radius + target_radius

		if distance_between_centers < max_distance:
			# hit
			reward = 100
			self.is_terminal = True
		else:
			reward = -distance_between_centers
		#print(reward)
		return reward

	def step(self, action):
		omega1 = actions[action][0]
		omega2 = actions[action][1]
		self.move_arm(omega1, omega2)
		reward = self.get_reward()
		next_state = self.state
		info = {}
		return next_state, reward, self.is_terminal, info

#######################################################################################