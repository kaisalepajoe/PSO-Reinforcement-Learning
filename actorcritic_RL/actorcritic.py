import torch
from torch import nn
from torch import optim
import numpy as np 
from torch.nn import functional as F 
from env3 import RobotArmGame
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
import time

def convert_to_tkinter_coords(state, env):
	# bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y
	for x in [0,2,4]:
		state[x] = env.window_width/2*(state[x] + 1)
	for y in [1,3,5]:
		state[y] = env.window_height/2*(1 - state[y])
	return state


# Create window
def create_window(window_width, window_height):
	window = tk.Tk()
	canvas = tk.Canvas(window, width=window_width, height=window_height, bg="white")
	canvas.grid() # tell tkinter where to place the canvas
	window.title("Robot Arm")
	return window, canvas

def get_corners(x, y, r):
	'''
	Returns the coordinates of the corners of a circle
	Parameters
	----------
	x (float): x-coordinate of center
	y (float): y-coordinate of center
	r (float): radius of circle
	Returns
	-------
	(x0, y0, x1, y1)
	x0 (float):x-coordinate of upper left corner
	y0 (float):y-coordinate of upper left corner
	x1 (float):x-coordinate of lower right corner
	y1 (float):y-coordinate of lower right corner
	'''
	x0 = x - r
	y0 = y - r
	x1 = x + r
	y1 = y + r
	return x0, y0, x1, y1

# Draw all shapes with initial positions
def initial_draw(canvas, initial_conditions_tk, env):
	bob1_angle = env.angles[0]
	bob2_angle = env.angles[1]
	# Initial conditions with coordinate 0 in centre of window
	bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y = initial_conditions_tk.astype(float)

	x_center = env.window_width/2
	y_center = env.window_height/2

	# Draw the first limb
	limb1 = canvas.create_line(x_center, y_center, bob1_x, bob1_y, width=env.line_width)

	# Draw the second limb
	limb2 = canvas.create_line(bob1_x, bob1_y, bob2_x, bob2_y, width=env.line_width)

	# Draw the first bob
	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(bob1_x, bob1_y, env.bob1_radius*200)
	bob1 = canvas.create_oval(bob1_x0, bob1_y0, bob1_x1, bob1_y1, fill="black")

	# Draw the second bob
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(bob2_x, bob2_y, env.bob2_radius*200)
	bob2 = canvas.create_oval(bob2_x0, bob2_y0, bob2_x1, bob2_y1, fill="blue")

	# Draw the centre pivot
	cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1 = get_corners(x_center,y_center,10)
	cpivot = canvas.create_oval(cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1, fill="black")

	# Draw the target
	target_x0, target_y0, target_x1, target_y1 = get_corners(target_x, target_y, env.target_radius*200)
	target = canvas.create_oval(target_x0, target_y0, target_x1, target_y1, fill="red")

	elements = {
		"limb1":limb1,
		"limb2":limb2,
		"bob1":bob1,
		"bob2":bob2,
		"target":target
	}
	return elements

# Move pendulum
def move(canvas, elements, next_state_tk, env):
	x_center = env.window_width/2
	y_center = env.window_height/2

	bob1_x, bob1_y, bob2_x, bob2_y, target_x, target_y = next_state_tk.astype(float)

	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(bob1_x, bob1_y, env.bob1_radius*200)
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(bob2_x, bob2_y, env.bob2_radius*200)
	target_x0, target_y0, target_x1, target_y1 = get_corners(target_x, target_y, env.target_radius*200)

	canvas.coords(elements["limb1"], x_center, y_center, bob1_x, bob1_y)
	canvas.coords(elements["limb2"], bob1_x, bob1_y, bob2_x, bob2_y)
	canvas.coords(elements["bob1"], bob1_x0, bob1_y0, bob1_x1, bob1_y1)
	canvas.coords(elements["bob2"], bob2_x0, bob2_y0, bob2_x1, bob2_y1)
	canvas.coords(elements["target"], target_x0, target_y0, target_x1, target_y1)

def animate(visited, env):
	window, canvas = create_window(env.window_width, env.window_height)
	initial_conditions_tk = convert_to_tkinter_coords(visited[0], env)
	elements = initial_draw(canvas, initial_conditions_tk, env)
	for t in range(1,len(visited)):
		state = visited[t]
		state_tk = convert_to_tkinter_coords(state, env)
		move(canvas, elements, state_tk, env)
		window.update()
		time.sleep(0.005)
	window.mainloop()

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

def run_episode(env, model, target_position):
	state = torch.from_numpy(env.reset(target_position)).float()
	values, logprobs, rewards = [], [], []
	visited = []
	done = False
	t = 0
	while not done:
		if t < 6000:
			visited.append(state.numpy())
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
	return values, logprobs, rewards, visited

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
	optimizer.step()
	return actor_loss, critic_loss, len(rewards)

epochs=200

env = RobotArmGame()
model = ActorCritic()
target_position = np.random.uniform(-1,1,2)

optimizer = optim.Adam(lr=1e-4, params=model.parameters())
optimizer.zero_grad()
total_rewards = []
for i in tqdm(range(epochs)):
	optimizer.zero_grad()
	values, logprobs, rewards, visited_states = run_episode(env, model, target_position)
	actor_loss, critic_loss, episode_len = update_params(optimizer, values, logprobs, rewards)
	total_rewards.append(np.sum(np.array(rewards)))

x = np.arange(len(total_rewards))
plt.plot(x,total_rewards)
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.show()

animate(visited_states, env)