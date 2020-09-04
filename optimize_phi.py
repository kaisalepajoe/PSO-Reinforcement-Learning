from train2 import train
from evaluate import get_scores
from evaluate import get_random_scores
import numpy as np
import matplotlib.pyplot as plt

phis = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 2.9, 3.0])
#phis = np.array([2.2,2.4,2.6])
training_repetitions = 3
training_time_steps = 2

average_results = np.inf*np.ones(len(phis))
random_results = np.inf*np.ones(len(phis))

for i,phi in enumerate(phis):
	print(f"Testing phi = {phi}")
	avg_scores = np.inf*np.ones(training_repetitions)
	for rep in range(training_repetitions):
		params, _ = train(phi=phi, time_steps=training_time_steps, plot=False, show_animation=False)
		scores = get_scores(params)
		avg_score = np.average(scores)
		avg_scores[rep] = avg_score
	avg_result = np.average(avg_scores)
	average_results[i] = avg_result
	random_scores = get_random_scores(len(params))
	avg_random_score = np.average(random_scores)
	random_results[i] = avg_random_score

plt.plot(phis, average_results, label='Trained')
plt.plot(phis, random_results, label='Random')
plt.title(f"Testing different values of phi with {training_time_steps} training time steps and averaged over {training_repetitions} training repetitions.")
plt.legend()
plt.xlabel(f"Phi")
plt.ylabel(f"Average time per 1000 episodes")
plt.show()