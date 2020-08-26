import numpy as np
from environment import RobotArmGame
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
import time

def model(x, unpacked_params):
	l1,b1,l2,b2 = unpacked_params
	y = torch.nn.functional.linear(x,l1,b1)
	y = torch.relu(y)
	y = torch.nn.functional.linear(y,l2,b2)
	y = torch.log_softmax(y,dim=0)
	return y

def unpack_params(params):
	hidden_layer = int((len(params) - 9)/14)
	params = torch.from_numpy(params).float()
	layers = [(hidden_layer,4), (9,hidden_layer)]
	unpacked_params = []
	e = 0
	for i,l in enumerate(layers):
		s,e = e,e+np.prod(l)
		weights = params[s:e].view(l)
		s,e = e,e+l[0]
		bias = params[s:e]
		unpacked_params.extend([weights,bias])
	return unpacked_params

def get_scores(vector, iterations=1000):
	env = RobotArmGame()
	scores = []
	params = unpack_params(vector)
	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset()).float()
		score = 0
		t = 0
		while not done:
			if t < 4000:

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

def get_random_scores(vector_length, iterations=1000):
	env = RobotArmGame()
	scores = []

	for iteration in tqdm(range(iterations)):
		vector = 20*np.random.random(vector_length) - 10
		params = unpack_params(vector)
		done = False
		state = torch.from_numpy(env.reset()).float()
		score = 0
		t = 0
		while not done:
			if t < 4000:

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

def compare_to_random(vector, iterations=1000):
	'''
	Compares the found NN weights and biases vector to a random one

	Input
	-----
	vector : np.ndarray
	A vector of length 177 that contains the weights and biases of the neural network

	iterations : int (default 1000)
	The number of times to run the game before plotting results
	'''
	optimal_scores = get_scores(vector, iterations)
	random_scores = get_random_scores(len(vector), iterations)
	optimal_points = []
	random_points = []
	for i in range(len(optimal_scores)):
		optimal_points.append(np.sum(optimal_scores[0:i]))
		random_points.append(np.sum(random_scores[0:i]))
	x = np.arange(len(optimal_points))
	plt.plot(x,optimal_points, label='learned')
	plt.plot(x,random_points, label='random')
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

# Get the x and y coordinate of the end of the limb (tkinter coords)
def get_limb_end(x0, y0, length, angle):
	x1 = x0 + length*np.cos(angle)
	y1 = y0 - length*np.sin(angle)
	return x1, y1

# Draw all shapes with initial positions
def initial_draw(canvas, initial_conditions, drawing_parameters):
	theta1 = initial_conditions[0]
	theta2 = initial_conditions[1]

	l1 = drawing_parameters["l1"]
	l2 = drawing_parameters["l2"]
	bob1_radius = drawing_parameters["bob1_radius"]
	bob2_radius = drawing_parameters["bob2_radius"]
	target_radius = drawing_parameters["target_radius"]
	window_width = drawing_parameters["window_width"]
	window_height = drawing_parameters["window_height"]
	x_center = window_width/2
	y_center = window_height/2
	line_width = drawing_parameters["line_width"]

	# Draw the first limb
	limb1_x1, limb1_y1 = get_limb_end(x_center, y_center, l1*200, theta1)
	limb1 = canvas.create_line(x_center, y_center, limb1_x1, limb1_y1, width=line_width)

	# Draw the second limb
	limb2_x1, limb2_y1 = get_limb_end(limb1_x1, limb1_y1, l2*200, theta2)
	limb2 = canvas.create_line(limb1_x1, limb1_y1, limb2_x1, limb2_y1, width=line_width)

	# Draw the first bob
	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(limb1_x1, limb1_y1, bob1_radius*200)
	bob1 = canvas.create_oval(bob1_x0, bob1_y0, bob1_x1, bob1_y1, fill="black")

	# Draw the second bob
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius*200)
	bob2 = canvas.create_oval(bob2_x0, bob2_y0, bob2_x1, bob2_y1, fill="blue")

	# Draw the centre pivot
	cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1 = get_corners(x_center,y_center,10)
	cpivot = canvas.create_oval(cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1, fill="black")

	# Draw the target
	target_x0, target_y0, target_x1, target_y1 = get_corners(initial_conditions[2]*200 + x_center, y_center - initial_conditions[3]*200, target_radius*200)
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
def move(canvas, elements, next_state, drawing_parameters):
	l1 = drawing_parameters["l1"]
	l2 = drawing_parameters["l2"]
	bob1_radius = drawing_parameters["bob1_radius"]
	bob2_radius = drawing_parameters["bob2_radius"]
	target_radius = drawing_parameters["target_radius"]
	window_width = drawing_parameters["window_width"]
	window_height = drawing_parameters["window_height"]
	x_center = window_width/2
	y_center = window_height/2
	line_width = drawing_parameters["line_width"]

	theta1 = next_state[0]
	theta2 = next_state[1]
	target_x = next_state[2]
	target_y = next_state[3]

	limb1_x1, limb1_y1 = get_limb_end(x_center, y_center, l1*200, theta1)
	limb2_x1, limb2_y1 = get_limb_end(limb1_x1, limb1_y1, l2*200, theta2)
	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(limb1_x1, limb1_y1, bob1_radius*200)
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius*200)
	target_x0, target_y0, target_x1, target_y1 = get_corners(target_x*200 + x_center, y_center - target_y*200, target_radius*200)

	canvas.coords(elements["limb1"], x_center, y_center, limb1_x1, limb1_y1)
	canvas.coords(elements["limb2"], limb1_x1, limb1_y1, limb2_x1, limb2_y1)
	canvas.coords(elements["bob1"], bob1_x0, bob1_y0, bob1_x1, bob1_y1)
	canvas.coords(elements["bob2"], bob2_x0, bob2_y0, bob2_x1, bob2_y1)
	canvas.coords(elements["target"], target_x0, target_y0, target_x1, target_y1)

	return

def animate(vector, time_steps=10_000):
	env = RobotArmGame()
	visited = np.zeros((10_000,4))
	params = unpack_params(vector)
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

	drawing_parameters = {
		"l1":0.5,
		"l2":0.5,
		"bob1_radius":0.1,
		"bob2_radius":0.1,
		"target_radius":0.2,
		"window_width":600,
		"window_height":600,
		"line_width":2
	}

	window, canvas = create_window(drawing_parameters["window_width"], drawing_parameters["window_height"])
	elements = initial_draw(canvas, visited[0,:], drawing_parameters)

	for t in range(1,time_steps):
		next_state = visited[t]
		move(canvas, elements, next_state, drawing_parameters)
		window.update()
		time.sleep(0.005)
	window.mainloop()

	return