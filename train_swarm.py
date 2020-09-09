# This algorithm is inspired by PSO(0) from the book 
# "Particle Swarm Optimization" by Maurice Clerc.
# The goal is to train a robotic arm to reach a target on the screen

###################################################################

# Import required modules
import numpy as np 
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pkgutil
import io
import sys
from tqdm import tqdm
import torch
from env2 import RobotArmGame
import tkinter as tk
import time

###################################################################

def unpack_params(params, layers):
	params = torch.from_numpy(params).float()
	unpacked_params = []
	e = 0
	for i,l in enumerate(layers):
		s,e = e,e+np.prod(l)
		weights = params[s:e].view(l)
		s,e = e,e+l[0]
		bias = params[s:e]
		unpacked_params.extend([weights,bias])
	return unpacked_params

def model(x, unpacked_params):
	y = torch.nn.functional.linear(x,unpacked_params[0],unpacked_params[1])
	for layer in range(1,int(len(unpacked_params)/2)):
		y = torch.relu(y)
		y = torch.nn.functional.linear(y,unpacked_params[layer*2],unpacked_params[layer*2+1])
	y = torch.tanh(y)
	return y

def epsilon_greedy(state, params, layers):
	epsilon = 0.01
	random_number = np.random.random()
	if random_number < epsilon:
		action = np.random.uniform(-1,1,2)
	else:
		action = model(state, params).numpy()
	return action

def get_target_distance(bob2_position, target_position):
	bob_x, bob_y = bob2_position
	target_x, target_y = target_position
	d = np.sqrt((bob_x - target_x)**2 + (bob_y - target_y)**2)
	return d

def get_min_time(target_distance, max_speed=0.05):
	return target_distance/max_speed

def evaluate(params_vector, layers, target_position, random_action=False):
	global n_evaluations
	n_evaluations += 1
	env = RobotArmGame()
	done = False
	state = torch.from_numpy(env.reset(target_position)).float()
	target_distance = get_target_distance(state[2:4], state[4:6])
	min_time = get_min_time(target_distance)
	t = 0
	total_reward = 0
	while not done:
		if t < 3000:
			if random_action == False:
				params = unpack_params(params_vector, layers)
				action = epsilon_greedy(state, params, layers)
			else:
				action = np.random.uniform(-1,1,2)
			state_, reward, done, info = env.step(action)
			state = torch.from_numpy(state_).float()
			t += 1
			total_reward += reward
		else:
			done = True
	# Testing scaling by minimum possible time
	return total_reward/min_time


def get_corners(x, y, r):
	'''
	Returns the coordinates of the corners of a circle
	Parameters
	----------
	x (float): x-coordinate of center
	y (float): y-coordinate of center
	r (float): radius of circle
	Returns
	-------
	(x0, y0, x1, y1)
	x0 (float):x-coordinate of upper left corner
	y0 (float):y-coordinate of upper left corner
	x1 (float):x-coordinate of lower right corner
	y1 (float):y-coordinate of lower right corner
	'''
	x0 = x - r
	y0 = y - r
	x1 = x + r
	y1 = y + r
	return x0, y0, x1, y1

def convert_to_tkinter_coords(state, env):
	# bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y
	for x in [0,2,4]:
		state[x] = env.window_width/2*(state[x] + 1)
	for y in [1,3,5]:
		state[y] = env.window_height/2*(1 - state[y])
	return state

# Create window
def create_window(window_width, window_height):
	window = tk.Tk()
	canvas = tk.Canvas(window, width=window_width, height=window_height, bg="white")
	canvas.grid() # tell tkinter where to place the canvas
	window.title("Robot Arm")
	return window, canvas

# Draw all shapes with initial positions
def initial_draw(canvas, initial_conditions_tk, env):
	bob1_angle = env.angles[0]
	bob2_angle = env.angles[1]
	# Initial conditions with coordinate 0 in centre of window
	bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y = initial_conditions_tk

	x_center = env.window_width/2
	y_center = env.window_height/2

	# Draw the first limb
	limb1 = canvas.create_line(x_center, y_center, bob1_x, bob1_y, width=env.line_width)

	# Draw the second limb
	limb2 = canvas.create_line(bob1_x, bob1_y, bob2_x, bob2_y, width=env.line_width)

	# Draw the first bob
	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(bob1_x, bob1_y, env.bob1_radius*200)
	bob1 = canvas.create_oval(bob1_x0, bob1_y0, bob1_x1, bob1_y1, fill="black")

	# Draw the second bob
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(bob2_x, bob2_y, env.bob2_radius*200)
	bob2 = canvas.create_oval(bob2_x0, bob2_y0, bob2_x1, bob2_y1, fill="blue")

	# Draw the centre pivot
	cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1 = get_corners(x_center,y_center,10)
	cpivot = canvas.create_oval(cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1, fill="black")

	# Draw the target
	target_x0, target_y0, target_x1, target_y1 = get_corners(target_x, target_y, env.target_radius*200)
	target = canvas.create_oval(target_x0, target_y0, target_x1, target_y1, fill="red")

	elements = {
		"limb1":limb1,
		"limb2":limb2,
		"bob1":bob1,
		"bob2":bob2,
		"target":target
	}
	return elements

# Move pendulum
def move(canvas, elements, next_state_tk, env):
	x_center = env.window_width/2
	y_center = env.window_height/2

	bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y = next_state_tk

	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(bob1_x, bob1_y, env.bob1_radius*200)
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(bob2_x, bob2_y, env.bob2_radius*200)
	target_x0, target_y0, target_x1, target_y1 = get_corners(target_x, target_y, env.target_radius*200)

	canvas.coords(elements["limb1"], x_center, y_center, bob1_x, bob1_y)
	canvas.coords(elements["limb2"], bob1_x, bob1_y, bob2_x, bob2_y)
	canvas.coords(elements["bob1"], bob1_x0, bob1_y0, bob1_x1, bob1_y1)
	canvas.coords(elements["bob2"], bob2_x0, bob2_y0, bob2_x1, bob2_y1)
	canvas.coords(elements["target"], target_x0, target_y0, target_x1, target_y1)

###################################################################

class Swarm():
	def __init__(self, info):
		self.N = info['N']
		self.time_steps = info['time_steps']
		self.repetitions = info['repetitions']
		self.k = info['k']
		self.dim = info['dim']
		self.search_space = info['search_space']
		self.layers = info['layers']
		self.disable_progress_bar = info['disable_progress_bar']
		self.final_result_from = info['final_result_from']

		self.xmin = np.ones(self.dim)*-1*self.search_space
		self.xmax = np.ones(self.dim)*self.search_space
		self.vmax = np.absolute(self.xmax - self.xmin)/2
		self.c1 = 0.7298
		self.cmax = 1.4960

		self.info = info

	def random_initial_positions(self):
		initial_positions = np.inf*np.ones((self.N, self.dim))
		for d in range(self.dim):
			initial_positions[:,d] = np.random.uniform(self.xmin[d], self.xmax[d], self.N)
		return initial_positions

	def random_initial_velocities(self):
		initial_velocities = np.random.uniform(-self.vmax, self.vmax, (self.N, self.dim))
		return initial_velocities

	def create_particles(self, initial_positions, initial_velocities, target_position):
		# Create array of initial p-values by evaluating initial positions
		p_values = np.inf*np.ones((self.N, self.dim+1))
		for i, pos in enumerate(initial_positions):
			p_values[i,0:self.dim] = pos		
			value = evaluate(pos, self.layers, target_position)
			p_values[i,self.dim] = value
		# Create list of particle objects
		self.particles = []
		for i in range(self.N):
			pos = initial_positions[i]
			vel = initial_velocities[i]
			p = p_values[i]
			particle = Particle(self.info)
			particle.set_initial_state(pos, vel, p)
			self.particles.append(particle)

	def random_informants(self):
		for particle in self.particles:
			particle.informants = np.random.choice(self.particles, particle.k)

	def distribute_swarm(self, target_position):
		# Create array of initial positions and velocities
		initial_positions = self.random_initial_positions()
		initial_velocities = self.random_initial_velocities()

		self.create_particles(initial_positions, initial_velocities, target_position)

		# Initiate k informants randomly
		self.random_informants()

	def evolve(self, target_position):
		# Initialise array of positions for animation
		self.positions = np.inf*np.ones((self.time_steps, self.N, self.dim+1))		
		# Evolve swarm for all time steps
		self.avg_rewards = []
		for time_step in tqdm(range(self.time_steps),
			desc=f"Repetition {self.current_repetition}/{self.repetitions}: Evolving swarm",
			disable=self.disable_progress_bar):
			particle_rewards = []
			for i, particle in enumerate(self.particles):
				particle_reward = particle.step(target_position)
				particle_rewards.append(particle_reward)
				# Update positions for animation
				self.positions[time_step,i,:-1] = particle.pos
				self.positions[time_step,i,-1] = particle_reward
			particle_rewards = np.array(particle_rewards)
			self.avg_rewards.append(np.average(particle_rewards))
			# Select informants for next time step
			self.random_informants()

	def get_parameters(self, target_position=None):
		if self.final_result_from == "g-values":
			final_g = np.inf*np.ones((self.N, self.dim+1))
			for i,particle in enumerate(self.particles):
				final_g[i,:] = particle.g
			optimal_i = np.argmax(final_g[:,self.dim])
			result = final_g[optimal_i]
		if self.final_result_from == "average p-values":
			final_p = np.inf*np.ones((self.N, self.dim+1))
			for i,particle in enumerate(self.particles):
				final_p[i,:] = particle.p 
			result = np.average(final_p, axis=0)
		if self.final_result_from == "average final pos":
			final_pos = np.inf*np.ones((self.N, self.dim+1))
			for i,particle in enumerate(self.particles):
				final_pos[i,:self.dim] = particle.pos
			result = np.average(final_pos, axis=0)
		if self.final_result_from == "centre of gravity":
			print('Calculating centre of gravity')
			final_pos = np.inf*np.ones((self.N, self.dim+1))
			repetitions = 10
			all_scores = np.inf*np.ones((self.N, repetitions))
			for rep in tqdm(range(repetitions)):
				for i, particle in enumerate(self.particles):
					final_pos[i,:self.dim] = particle.pos 
					score = evaluate(particle.pos, particle.layers, target_position)
					all_scores[i,rep] = -1*score
			average_scores = np.average(all_scores, axis=1)
			avg_scores_matrix = np.inf*np.ones((self.N, self.dim+1))
			for column in range(self.dim+1):
				avg_scores_matrix[:,column] = average_scores
			numerator = np.sum(final_pos*avg_scores_matrix, axis=0)
			result = numerator/np.sum(average_scores)
		return result

	def run_algorithm(self, target_position):
		results = np.inf*np.ones((self.repetitions, self.dim+1))
		# all_positions contains all the visited positions for each repetition
		# all_positions is used to create an animation of the swarm
		self.all_positions = np.inf*np.ones((self.repetitions, self.time_steps, self.N, self.dim+1))

		for r in range(self.repetitions):
			self.current_repetition = r+1
			self.distribute_swarm(target_position)
			self.evolve(target_position)
			result = self.get_parameters(target_position=target_position)
			results[r] = result
			self.all_positions[r] = self.positions

		self.best_value_index = np.argmax(results[:,self.dim])
		self.best_position = results[self.best_value_index][0:self.dim]
		self.best_average_scores = np.average(self.all_positions[self.best_value_index,:,:,-1], axis=1)
		assert self.best_average_scores.shape[0] == self.time_steps

		return self.best_position

	def animate_swarm(self):
		'''
		Plots an animation of the best repetition of evolving the swarm.
		Only the first 2 dimensions are plotted for a higher-dimensional problem.
		'''

		# Plot initial positions of particles
		fig, ax = plt.subplots()
		ax.set_xlim(self.xmin[0], self.xmax[0])
		ax.set_ylim(self.xmin[1], self.xmax[1])
		scat = ax.scatter(self.all_positions[self.best_value_index,0,:,0], 
			self.all_positions[self.best_value_index,0,:,1], color="Black", s=2)

		# Create animation
		interval = 200_000 / (self.N * self.time_steps * self.repetitions)
		self.animation = FuncAnimation(fig, func=self.update_frames, interval=interval, 
			fargs=[scat, self.all_positions, self.best_value_index])
		plt.show()

	def update_frames(self, j, *fargs):
		scat, all_positions, best_value_index = fargs
		try:
			scat.set_offsets(all_positions[best_value_index,j,:,0:2])
		except:
			print("Animation finished.")
			self.animation.event_source.stop()

###################################################################

# Particle objects are created within the swarm class methods. 
class Particle(Swarm):
	'''
	Particle instances are created within the Swarm class methods. Particles
	inherit constants and function info from the Experiment class.
	'''

	def set_initial_state(self, pos, vel, p):
		self.pos = pos
		self.vel = vel
		# Set initial best found value by particle
		# format: np array of shape (1, 3) - x, y, value
		self.p = p
		# Best found position and value by informants or itself
		# format: np array of shape (1, 3) - x, y, value
		self.g = p
		# Empty list of informants
		self.informants = []

	def communicate(self):
		'''
		Receives g-values from informants and updates the Particle's g-value accordingly.
		If the best received g-value is smaller than the Particle's g-value, then the
		particles g-value is set to the received g-value.
		'''
		# Receive best positions with values from informants
		received = np.zeros((self.k, self.dim+1))
		for i, informant in enumerate(self.informants):
			received[i, :] = informant.g
		# Find best g from communicated values
		i = np.argmax(received[:,self.dim])
		best_received_g = received[i]
		# Set g to LOWEST value
		if best_received_g[-1] > self.g[-1]:
			self.g = best_received_g

	def random_confidence(self):
		'''
		Randomly assigns confidence parameters c2 and c3 in the interval [0, cmax)
		'''
		c2 = np.inf*np.ones(self.dim)
		c3 = np.inf*np.ones(self.dim)

		for d in range(self.dim):
			c2[d] = np.random.uniform(0, self.cmax)
			c3[d] = np.random.uniform(0, self.cmax)
		return (c2, c3)

	def update_p(self, value):
		if value > self.p[self.dim]:
			self.p[self.dim] = value
			self.p[0:self.dim] = self.pos

	def update_g(self, value):
		if value > self.g[self.dim]:
			self.g[self.dim] = value
			self.g[0:self.dim] = self.pos
		self.communicate()

	def find_vel(self):
		c2, c3 = self.random_confidence()		
		possible_vel = self.c1*self.vel + \
			c2*(self.p[0:self.dim] - self.pos) + \
			c3*(self.g[0:self.dim] - self.pos)	

		# Constrain velocity
		smaller_than_vmax = possible_vel < self.vmax
		possible_vel = np.where(smaller_than_vmax, possible_vel, self.vmax)
		greater_than_neg_vmax = possible_vel > -self.vmax
		possible_vel = np.where(greater_than_neg_vmax, possible_vel, -self.vmax)
		self.vel = possible_vel
	
	def set_pos(self):
		possible_pos = self.pos + self.vel
		in_search_area = self.is_in_search_area(possible_pos)
		self.pos = np.where(in_search_area, possible_pos, self.pos)
		self.vel = np.where(in_search_area, self.vel, 0)

	def is_in_search_area(self, possible_pos):
		smaller_than_xmax = possible_pos <= self.xmax
		greater_than_xmin = possible_pos >= self.xmin
		is_allowed = np.zeros((len(self.xmax), 2))
		is_allowed[:,0] = smaller_than_xmax
		is_allowed[:,1] = greater_than_xmin
		is_allowed = np.all(is_allowed, axis=1)
		return is_allowed

	def step(self, target_position):
		value = evaluate(self.pos, self.layers, target_position)
		self.update_p(value)
		self.update_g(value)
		self.find_vel()
		self.set_pos()
		return value

###################################################################

class Training():
	def __init__(self, hidden_layers=[15,10], search_space=2,
		N=9, time_steps=100, repetitions=1, k=3, final_result_from='average final pos',
		show_animation=True, disable_progress_bar=False, disable_printing=True, plot=True):
		
		global n_evaluations
		n_evaluations = 0

		layers = []
		for layer in range(len(hidden_layers)+1):
			if layer == 0:
				layers.append((hidden_layers[0],6))
			elif layer == len(hidden_layers):
				layers.append((2,hidden_layers[-1]))
			else:
				layers.append((hidden_layers[layer], hidden_layers[layer-1]))
		self.layers = layers

		vector_length = 0
		for layer in layers:
			vector_length += layer[0]*layer[1] + layer[0]
		self.vector_length = vector_length

		self.info = {
			'N':N,
			'time_steps':time_steps,
			'repetitions':repetitions,
			'k':k,
			'dim':vector_length,
			'search_space':search_space,
			'layers':layers,
			'final_result_from':final_result_from,
			'disable_progress_bar':disable_progress_bar,
		}

		self.target_position = np.random.uniform(-1,1,2)

		print(f"Vector length: {self.vector_length}")
		print(f"Layers: {hidden_layers}")
		# Create swarm and evolve the swarm for the required number of repetitions.
		swarm = Swarm(self.info)
		swarm.distribute_swarm(self.target_position)
		swarm.run_algorithm(self.target_position)
		self.params_vector = swarm.best_position

		if disable_printing == False:
			print(f"{n_evaluations} evaluations made")

		if show_animation == False:
			pass
		else:
			swarm.animate_swarm()

		if plot == True:
			plt.plot(np.arange(1,time_steps+1), swarm.best_average_scores)
			plt.xlabel('Time step')
			plt.ylabel('Average score of swarm')
			plt.show()


	def animate(self, time_steps=10_000):
		env = RobotArmGame()
		visited = np.zeros((time_steps,6))
		params = unpack_params(self.params_vector, layers=self.layers)
		done = False
		state = torch.from_numpy(env.reset(self.target_position)).float()
		for t in tqdm(range(time_steps)):
			visited[t] = state.numpy()
			action = epsilon_greedy(state, params, self.layers)
			state_, reward, done, info = env.step(action)
			if done == True:
				state_ = env.reset(self.target_position)
			state = torch.from_numpy(state_).float()
		window, canvas = create_window(env.window_width, env.window_height)
		initial_conditions_tk = convert_to_tkinter_coords(visited[0], env)
		elements = initial_draw(canvas, initial_conditions_tk, env)

		for t in range(1,time_steps):
			next_state = visited[t]
			next_state_tk = convert_to_tkinter_coords(next_state, env)
			move(canvas, elements, next_state_tk, env)
			window.update()
			time.sleep(0.005)
		window.mainloop()