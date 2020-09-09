import numpy as np
from env2 import RobotArmGame
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
import time
from train_swarm import model
from train_swarm import unpack_params

def get_scores(vector, layers, target_position, iterations=1000):
	env = RobotArmGame()
	scores = []
	params = unpack_params(vector, layers=layers)
	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset(target_position)).float()
		score = 0
		t = 0
		while not done:
			if t < 3000:

				action = model(state, params)
				state_, reward, done, info = env.step(action.numpy())
				state = torch.from_numpy(state_).float()
				t += 1
			else:
				done = True
		score = t
		scores.append(score)
	scores = np.array(scores)
	return scores

def get_random_scores(vector_length, layers, target_position, iterations=1000, search_space = 10):
	env = RobotArmGame()
	scores = []

	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset(target_position)).float()
		score = 0
		t = 0
		params = np.random.uniform(-1,1,vector_length)
		params = unpack_params(params, layers)
		while not done:
			if t < 3000:
				action = model(state, params)
				state_, reward, done, info = env.step(action.numpy())
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
			if t < 3000:
				action = np.random.uniform(-0.05, 0.05, 2)
				state_, reward, done, info = env.step(action)
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




	return