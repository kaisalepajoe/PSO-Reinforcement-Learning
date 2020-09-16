# From scratch robot arm environment
import numpy as np

speed = 0.01
actions = np.array([
	[-speed, -speed],
	[-speed, 0],
	[-speed, speed],
	[0, -speed],
	[0, speed],
	[speed, -speed],
	[speed, 0],
	[speed, speed],
	])

def generate_random_angle():
	# The angle is in radians
	angle = np.random.random()*2*np.pi
	return angle

def get_reward_ratio(state):
	bob_x, bob_y = state[2:4]
	target_x, target_y = state[4:6]
	d = np.sqrt((bob_x - target_x)**2 + (bob_y - target_y)**2)
	max_distance = 1
	ratio = d/max_distance
	return ratio

class RobotArmGame():
	def __init__(self, animate=False):
		self.window_width = 600
		self.window_height = 600
		self.bob1_radius = 0.05
		self.bob2_radius = 0.05
		self.target_radius = 0.05
		self.l1 = 0.25
		self.l2 = 0.25
		self.line_width = 2
		self.animate = animate

	def reset(self):
		target_x = 0.5
		target_y = 0.9

		bob1_angle = generate_random_angle()
		bob2_angle = generate_random_angle()

		bob1_x = 0.5 + self.l1*np.cos(bob1_angle)
		bob1_y = 0.5 + self.l1*np.sin(bob1_angle)

		bob2_x = bob1_x + self.l2*np.cos(bob2_angle)
		bob2_y = bob1_y + self.l2*np.sin(bob2_angle)

		self.angles = np.array([bob1_angle, bob2_angle])
		self.state = np.array([bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y])
		self.reward_ratio = get_reward_ratio(self.state)
		return self.state

	def get_reward(self):
		bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y = self.state
		distance_between_centers = np.sqrt((bob2_x - target_x)**2 + (bob2_y - target_y)**2)
		if distance_between_centers <= self.bob2_radius + self.target_radius:
			reward = 0
			done = True
		else:
			reward = -1
			done = False
		reward = reward/self.reward_ratio
		return reward, done

	def step(self, action_index):
		bob1_angle = self.angles[0] + actions[action_index][0]
		bob2_angle = self.angles[1] + actions[action_index][0] + actions[action_index][1]

		bob1_x = 0.5 + self.l1*np.cos(bob1_angle)
		bob1_y = 0.5 + self.l1*np.sin(bob1_angle)

		bob2_x = bob1_x + self.l2*np.cos(bob2_angle)
		bob2_y = bob1_y + self.l2*np.sin(bob2_angle)

		self.angles = np.array([bob1_angle, bob2_angle])
		self.state[0:4] = np.array([bob1_x, bob1_y, bob2_x, bob2_y])

		reward, done = self.get_reward()
		info = {}

		return self.state, reward, done, info