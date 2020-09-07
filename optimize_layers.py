import numpy as np
import matplotlib.pyplot as plt
from train2 import train
from evaluate import get_scores
from evaluate import get_random_scores

architectures = np.array([
	[150,50],
	[250,150],
	[600],
	[300],
	[600,150,100],
	[300,300,300],
	])

plot_x = np.arange(len(architectures))
plot_y_trained = np.zeros(len(architectures))
plot_y_random = np.zeros(len(architectures))

for i,hidden_layers in enumerate(architectures):
	print(f"Hidden layers: {hidden_layers}")
	params, layers = train(time_steps=100, hidden_layers=hidden_layers, show_animation=False, plot=False)
	trained_scores = get_scores(params, layers)
	random_scores = get_random_scores(len(params), layers)
	average_trained_score = np.average(trained_scores)
	average_random_score = np.average(random_scores)
	plot_y_trained[i] = average_trained_score
	plot_y_random[i] = average_random_score
	print(f"Average trained score: {average_trained_score}")
	print(f"Average random score: {average_random_score}")

plt.plot(plot_x, plot_y_trained, label='Trained')
plt.plot(plot_x, plot_y_random, label='Random')
plt.xlabel('Layer index')
plt.ylabel('Average time per episode')
plt.title('Testing different layer architectures')
plt.legend()