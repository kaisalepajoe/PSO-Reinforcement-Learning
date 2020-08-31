# Optimizing size of swarm
from modified_train import train
from modified_evaluate import get_scores
from modified_evaluate import get_random_scores
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

Ns = np.array([4,6,8,10,12,15,20,25,30,35,40,50,60,70,100])
repetitions = 10
params_times = np.zeros(len(Ns))
random_times = np.zeros(len(Ns))

for i,N in tqdm(enumerate(Ns)):
	repeated_params_times = np.zeros(repetitions)
	for repetition in range(repetitions):
		params, _ = train(N=N, time_steps=100, show_animation=False, disable_progress_bar=True)
		params_time = np.sum(get_scores(params, iterations=100))
		repeated_params_times[repetition] = params_time
	params_time = np.average(repeated_params_times)
	random_time = np.sum(get_random_scores(len(params), iterations=100))
	params_times[i] = params_time
	random_times[i] = random_time

plt.plot(Ns, params_times, label='trained')
plt.plot(Ns, random_times, label='random')
plt.title('Optimizing N, averaging over repetitions')
plt.xlabel('Number of particles in swarm')
plt.legend()
plt.show()

print(f"Trained times: {params_times}")
print(f"Random times: {random_times}")
