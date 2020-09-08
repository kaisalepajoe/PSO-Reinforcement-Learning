# From scratch robot arm environment
import numpy as np

def generate_random_angle():
	# The angle is in radians
	angle = np.random.random()*2*np.pi
	return angle

class RobotArmGame():
	def __init__(self, animate=False):
		self.window_width = 600
		self.window_height = 600
		self.bob1_radius = 0.1
		self.bob2_radius = 0.1
		self.target_radius = 0.1
		self.l1 = 0.5
		self.l2 = 0.5
		self.line_width = 2
		self.animate = animate

	def reset(self):
		# Generate initial state with constant target position and random arm position
		target_x = 0.25
		target_y = 0.75

		bob1_angle = generate_random_angle()
		bob2_angle = generate_random_angle()

		bob1_x = self.l1*np.cos(bob1_angle)
		bob1_y = self.l1*np.sin(bob1_angle)

		bob2_x = bob1_x + self.l2*np.cos(bob2_angle)
		bob2_y = bob1_y + self.l2*np.sin(bob2_angle)

		self.angles = np.array([bob1_angle, bob2_angle])
		self.state = np.array([bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y])

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
		return reward, done

	def step(self, action):
		max_speed = 0.05
		bob1_angle = self.angles[0] + action[0]*max_speed
		bob2_angle = self.angles[1] + action[1]*max_speed

		bob1_x = self.l1*np.cos(bob1_angle)
		bob1_y = self.l1*np.sin(bob1_angle)

		bob2_x = bob1_x + self.l2*np.cos(bob2_angle)
		bob2_y = bob1_y + self.l2*np.sin(bob2_angle)

		self.angles = np.array([bob1_angle, bob2_angle])
		self.state[0:4] = np.array([bob1_x, bob1_y, bob2_x, bob2_y])

		reward, done = self.get_reward()
		info = {}

		return self.state, reward, done, info