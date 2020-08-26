# Optimizing different parameters
from train import train
from evaluate import get_scores
from evaluate import get_random_scores
import matplotlib.pyplot as plt
import numpy as np

nodes = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
params_times = np.zeros(len(nodes))
random_times = np.zeros(len(nodes))

for i,node in enumerate(nodes):
	params, _ = train(time_steps=200, nodes=node, show_animation=False)
	params_time = np.sum(get_scores(params, iterations=500))
	random_time = np.sum(get_random_scores(len(params), iterations=500))
	params_times[i] = params_time
	random_times[i] = random_time

plt.plot(nodes, params_times, label='trained')
plt.plot(nodes, random_times, label='random')
plt.legend()
plt.show()

print(f"Trained times: {params_times}")
print(f"Random times: {random_times}")
