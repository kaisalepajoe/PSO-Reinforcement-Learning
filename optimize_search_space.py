# Optimizing the search space
from train import train
from evaluate import get_scores
from evaluate import get_random_scores
import matplotlib.pyplot as plt
import numpy as np

limits = np.array([1,2,3,4,5,6,7,8,9,10])
params_times = np.zeros(len(limits))
random_times = np.zeros(len(limits))

for i,limit in enumerate(limits):
	params, _ = train(time_steps=500, search_space=limit, show_animation=False)
	params_time = np.sum(get_scores(params, iterations=500))
	random_time = np.sum(get_random_scores(len(params), iterations=500))
	params_times[i] = params_time
	random_times[i] = random_time

plt.plot(limits, params_times, label='trained')
plt.plot(limits, random_times, label='random')
plt.xlabel('Search space limit')
plt.legend()
plt.show()

print(f"Finished training with different limits")
print(f"Trained times: {params_times}")
print(f"Random times: {random_times}")
