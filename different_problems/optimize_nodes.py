# Optimizing different parameters
from train import train
from evaluate import get_scores
from evaluate import get_random_scores
import matplotlib.pyplot as plt
import numpy as np

nodes = np.array([4,10,50,100,150])
reps = 5
params_times = np.zeros(len(nodes))
random_times = np.zeros(len(nodes))

for i,node in enumerate(nodes):
	repetition_params_times = np.zeros(reps)
	for r in range(reps):
		params, _ = train(time_steps=500, nodes=node, show_animation=False)
		params_time = np.sum(get_scores(params, iterations=500))
		repetition_params_times[r] = params_time
	params_time = np.average(repetition_params_times)
	random_time = np.sum(get_random_scores(len(params), iterations=500))
	params_times[i] = params_time
	random_times[i] = random_time

plt.plot(nodes, params_times, label='trained')
plt.plot(nodes, random_times, label='random')
plt.xlabel('Number of hidden nodes')
plt.legend()
plt.show()

print(f"Trained times: {params_times}")
print(f"Random times: {random_times}")
