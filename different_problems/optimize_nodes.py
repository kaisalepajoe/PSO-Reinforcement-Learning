# Optimizing size of swarm
from modified_train import train
from modified_evaluate import get_scores
from modified_evaluate import get_random_scores
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

nodes = np.array([150,100,80,60,40,30,20,18,16,14,12,10,8,6,4])
repetitions = 5
params_times = np.zeros(len(nodes))
random_times = np.zeros(len(nodes))

for i,node in enumerate(nodes):
	print(f'Testing nodes = {node}')
	repeated_params_times = np.zeros(repetitions)
	for repetition in range(repetitions):
		params, _ = train(N=9, nodes=node, time_steps=50, show_animation=False, disable_progress_bar=True, plot=False)
		params_time = np.sum(get_scores(params, iterations=100))
		repeated_params_times[repetition] = params_time
	params_time = np.average(repeated_params_times)
	random_time = np.sum(get_random_scores(len(params), iterations=100))
	params_times[i] = params_time
	random_times[i] = random_time

plt.plot(nodes, params_times, label='trained')
plt.plot(nodes, random_times, label='random')
plt.title('Optmizing number of hidden nodes, averaging over repetitions')
plt.xlabel('Number of nodes in hidden layer')
plt.ylabel('Accumulated score')
plt.legend()
plt.show()

print(f"Trained times: {params_times}")
print(f"Random times: {random_times}")
