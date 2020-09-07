import numpy as np
from env2 import RobotArmGame
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
import time
from train2 import model
from train2 import unpack_params

def get_scores(vector, layers, iterations=1000):
	env = RobotArmGame()
	scores = []
	params = unpack_params(vector, layers=layers)
	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset()).float()
		score = 0
		t = 0
		while not done:
			if t < 5000:

				probs = model(state, params)
				action = torch.distributions.Categorical(probs=probs).sample()
				state_, reward, done, info = env.step(action.item())
				state = torch.from_numpy(state_).float()
				t += 1
			else:
				done = True
		score = t
		scores.append(score)
	scores = np.array(scores)
	return scores

def get_random_scores(vector_length, layers, iterations=1000, search_space = 10):
	env = RobotArmGame()
	scores = []

	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset()).float()
		score = 0
		t = 0
		params = np.random.uniform(-1,1,vector_length)
		params = unpack_params(params, layers)
		while not done:
			if t < 5000:
				probs = model(state, params)
				action = torch.distributions.Categorical(probs=probs).sample()
				state_, reward, done, info = env.step(action.item())
				state = torch.from_numpy(state_).float()
				t += 1
			else:
				done = True
		score = t
		scores.append(score)
	scores = np.array(scores)
	return scores	

def get_random_net_scores(vector_length, layers, iterations=1000, search_space = 10):
	env = RobotArmGame()
	scores = []

	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset()).float()
		score = 0
		t = 0
		while not done:
			if t < 5000:
				action = np.random.choice(np.arange(8))
				state_, reward, done, info = env.step(action.item())
				state = torch.from_numpy(state_).float()
				t += 1
			else:
				done = True
		score = t
		scores.append(score)
	scores = np.array(scores)
	return scores	

def compare_to_random(vector, layers, iterations=1000):
	'''
	Compares the found NN weights and biases vector to a random one

	Input
	-----
	vector : np.ndarray
	A vector of length 177 that contains the weights and biases of the neural network

	iterations : int (default 1000)
	The number of times to run the game before plotting results
	'''
	optimal_scores = get_scores(vector, layers, iterations)
	random_scores = get_random_scores(len(vector), layers, iterations)
	random_net_scores = get_random_net_scores(len(vector), layers, iterations)
	optimal_points = []
	random_points = []
	random_net_points = []
	for i in range(len(optimal_scores)):
		optimal_points.append(np.sum(optimal_scores[0:i]))
		random_points.append(np.sum(random_scores[0:i]))
		random_net_points.append(np.sum(random_net_scores[0:i]))
	x = np.arange(len(optimal_points))
	plt.plot(x,optimal_points, label='learned')
	plt.plot(x,random_points, label='random')
	plt.plot(x,random_net_points, label='random weights')
	plt.xlabel('Iterations')
	plt.ylabel('Total time taken')
	plt.legend()
	plt.show()

# Create window
def create_window(window_width, window_height):
	window = tk.Tk()
	canvas = tk.Canvas(window, width=window_width, height=window_height, bg="white")
	canvas.grid() # tell tkinter where to place the canvas
	window.title("Robot Arm")
	return window, canvas

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

def animate(vector, layers, time_steps=10_000):
	env = RobotArmGame()
	visited = np.zeros((time_steps,6))
	params = unpack_params(vector, layers=layers)
	done = False
	state = torch.from_numpy(env.reset()).float()
	for t in tqdm(range(time_steps)):
		visited[t] = state.numpy()
		probs = model(state, params)
		action = torch.distributions.Categorical(probs=probs).sample()
		state_, reward, done, info = env.step(action.item())
		if done == True:
			state = torch.from_numpy(env.reset()).float()
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

	return