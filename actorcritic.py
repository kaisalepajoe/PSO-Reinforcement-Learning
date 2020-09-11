import torch
from torch import nn
from torch import optim
import numpy as np 
from torch.nn import functional as F 
from env2 import RobotArmGame

speed = 0.02
actions = np.array(
	[-speed, -speed],
	[-speed, 0],
	[-speed, speed],
	[0, -speed],
	[0, speed],
	[speed, -speed],
	[speed, 0],
	[speed, speed],
	)

class ActorCritic(nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()
		self.l1 = nn.Linear(6,25)
		self.l2 = nn.Linear(25,50)
		self.actor_lin1 = nn.Linear(50,8)
		self.l3 = nn.Linear(50,25)
		self.critic_lin1 = nn.Linear(25,1)
	def forward(self,x):
		x = F.normalize(x,dim=0)
		y = F.relu(self.l1(x))
		y = F.relu(self.l2(y))
		actor = F.log_softmax(self.actor_lin1(y),dim=0)
		c = F.relu(self.l3(y.detach()))
		critic = torch.tanh(self.critic_lin1(c))
		return actor, critic

def run_episode(env, model):
	target_position = np.random.uniform(-1,1,2)
	state = torch.from_numpy(env.reset(target_position)).float()
	values, logprobs, rewards = []
	done = False
	t = 0
	target_distance = get_target_distance(state[2:4], state[4:6])
	max_target_distance = 2
	distance_ratio = target_distance/max_target_distance
	while not done:
		if t < 6000:
			policy, value = model(state)
			values.append(value)
			logits = policy.view(-1)
			action_dist = torch.distributions.Categorical(logits=logits)
			action = action_dist.sample()
			logprob_ = policy.view(-1)[action]
			logprobs.append(logprob_)
			state_, reward, done, info = env.step(action.detach().numpy())
			state = torch.from_numpy(state_).float()
			t += 1
			rewards.append(reward)
		else:
			done = True
	return values, logprobs, rewards

def update_params(optimizer, values, logprobs, rewards, clc=0.1, gamma=0.95):
	rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
	logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
	values = torch.stack(values).flip(dims=(0,)).view(-1)
	returns = []
	ret_ = torch.Tensor([0])
	for r in range(rewards.shape[0]):
		ret_ = rewards[r] + gamma * ret_
		returns.append(ret_)
	returns = torch.stack(returns).view(-1)
	returns = F.normalize(returns,dim=0)
	actor_loss = -1*logprobs*(returns-values.detach())
	critic_loss = torch.pow(values-returns,2)
	loss = actor_loss.sum() + clc*critic_loss.sum()
	loss.backward()
	otpimizer.step()
	return actor_loss, critic_loss, len(rewards)

params = {'epochs':1000, 'n_workers':7}

env = RobotArmGame()
model = ActorCritic()

optimizer = optim.Adam(lr=1e-4, params=params)
optimizer.zero_grad()
for i in range(params['epochs']):
	optimizer.zero_grad()
	values, logprobs, rewards = run_episode(env, model)
	actor_loss, critic_loss, episode_len = update_params(optimizer, values, logprobs, rewards)
	print(episode_len)

