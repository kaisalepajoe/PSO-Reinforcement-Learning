import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
import time
import gym

def model(x, unpacked_params):
	l1,b1,l2,b2 = unpacked_params
	y = torch.nn.functional.linear(x,l1,b1)
	y = torch.relu(y)
	y = torch.nn.functional.linear(y,l2,b2)
	y = torch.log_softmax(y,dim=0)
	return y

def unpack_params(params):
	hidden_layer = int((len(params) - 6)/10)
	params = torch.from_numpy(params).float()
	layers = [(hidden_layer,6), (3,hidden_layer)]
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
	env = gym.make('Acrobot-v1')
	scores = []
	params = unpack_params(vector)
	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset()).float()
		total_reward = 0
		t = 0
		while not done:
			if t < 8000:

				probs = model(state, params)
				action = torch.distributions.Categorical(probs=probs).sample()
				state_, reward, done, info = env.step(action.item())
				state = torch.from_numpy(state_).float()
				t += 1
				total_reward += reward
			else:
				done = True
		score = total_reward
		scores.append(score)
	scores = np.array(scores)
	return scores

def get_random_scores(vector_length, iterations=1000):
	env = gym.make('Acrobot-v1')
	scores = []

	for iteration in tqdm(range(iterations)):
		vector = 20*np.random.random(vector_length) - 10
		params = unpack_params(vector)
		done = False
		state = torch.from_numpy(env.reset()).float()
		score = 0
		t = 0
		total_reward = 0
		while not done:
			if t < 8000:

				probs = model(state, params)
				action = torch.distributions.Categorical(probs=probs).sample()
				state_, reward, done, info = env.step(action.item())
				state = torch.from_numpy(state_).float()
				t += 1
				total_reward += reward
			else:
				done = True
		score = total_reward
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
	plt.ylabel('Total reward')
	plt.legend()
	plt.show()

def animate(vector, iterations=10_000):
	env = gym.make('Acrobot-v1')
	params = unpack_params(vector)
	for iteration in tqdm(range(iterations)):
		done = False
		state = torch.from_numpy(env.reset()).float()
		while not done:
				probs = model(state, params)
				action = torch.distributions.Categorical(probs=probs).sample()
				state_, reward, done, info = env.step(action.item())
				state = torch.from_numpy(state_).float()
				env.render()