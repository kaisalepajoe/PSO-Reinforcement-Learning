import gym
import torch
import numpy as np 
import matplotlib.pyplot as plt
from environment import RobotArmGame
from tqdm import tqdm

env = PendulumGame()

layer1 = 4
layer2 = 150
layer3 = 9 # There are 9 different actions in the pendulum game
MAX_DUR = 4000
MAX_EPISODES = 100_000
gamma = 0.99
score = []

speed = 0.05
actions = np.array([[-speed,-speed],
					[-speed,0],
					[-speed,speed],
					[0,-speed],
					[0,0],
					[0,speed],
					[speed,-speed],
					[speed,0],
					[speed,speed]])

model = torch.nn.Sequential(
	torch.nn.Linear(layer1,layer2),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(layer2,layer3),
	torch.nn.Softmax()
	)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def discount_rewards(rewards, gamma=0.99):
	lenr = len(rewards)
	disc_return = torch.pow(gamma, torch.arange(lenr).float())*rewards
	disc_return /= disc_return.max()
	return disc_return

def loss_fn(preds,r):
	return -1*torch.sum(r*torch.log(preds))

try:
	checkpoint = torch.load('./friday_statedict.pt')
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Loaded network")
except:
	print("No network to load")

for episode in tqdm(range(MAX_EPISODES)):
	curr_state = env.reset() # returns 4D array
	is_terminal = False
	transitions = []

	for t in range(MAX_DUR):
		act_prob = model(torch.from_numpy(curr_state).float())
		action = np.random.choice(np.arange(len(actions)), p=act_prob.data.numpy())
		prev_state = curr_state
		curr_state, reward, is_terminal, info = env.step(action) # this time we need reward explicitly
		transitions.append((prev_state, action, reward))
		if is_terminal:
			break

	ep_len = len(transitions)
	score.append(ep_len)
	reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))
	disc_rewards = discount_rewards(reward_batch)
	state_batch = torch.Tensor([s for (s,a,r) in transitions])
	action_batch = torch.Tensor([a for (s,a,r) in transitions])
	pred_batch = model(state_batch)
	prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
	loss = loss_fn(prob_batch, disc_rewards)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

try:
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict()},
		'friday_statedict.pt')
	print("Saved!")	
except:
	pass

print(np.average(np.array(score)))
number = np.arange(len(score))
plt.plot(number,score)
plt.show()