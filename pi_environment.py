import numpy as np 
import random
import tkinter as tk 
import time
from scipy.integrate import odeint
import torch

#####################################################

# Used resources

# Inspiration for how to solve Euler Lagrange https://scipython.com/blog/the-double-pendulum/
# Tutorial for scipy's odeint https://www.youtube.com/watch?v=VV3BnroVjZo
# Book: Deep Reinforcement Learning in Action

#####################################################

# Set drawing parameters
window_width = 600
window_height = 600
x_center = window_width/2
y_center = window_height/2
l1 = 100
l2 = 100
linewidth = 2
bob1_radius = 10
bob2_radius = 10
target_radius = 28

# Set neural network parameters
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
layer1 = 4
layer2 = 150
layer3 = 100
layer4 = 9
learning_rate = 0.0009
gamma = 0.99

training_sessions = 6000
maximum_batch_size = 6000

hit_time_steps = []
#np.random.seed(123)

#######################################################################################

# Create window
def create_window():
	window = tk.Tk()
	canvas = tk.Canvas(window, width=window_width, height=window_height, bg="#FFFFFC")
	canvas.grid() # tell tkinter where to place the canvas
	window.title("Robot arm environment")
	return window, canvas

def get_corners(x, y, r):
	'''
	Returns the coordinates of the corners of a circle
	'''
	x0 = x - r
	y0 = y - r
	x1 = x + r
	y1 = y + r
	return x0, y0, x1, y1

# Get the x and y coordinate of the end of the limb
def get_limb_end(x0, y0, length, angle):
	x1 = x0 + length*np.sin(angle)
	y1 = y0 + length*np.cos(angle)
	return x1, y1

def generate_initial_conditions():
	theta1 = generate_random_angle()
	theta2 = generate_random_angle()

	return [theta1, theta2]

def generate_random_angle():
	'''
	Returns a random angle.
	'''
	random_angle = np.random.random()*2*np.pi
	return random_angle

#######################################################################################

class PendulumGame():
	def __init__(self):

		self.window, self.canvas = create_window()

		self.initial_conditions = generate_initial_conditions()
		self.hits = 0

		self.initial_draw(self.canvas, self.initial_conditions)

		self.state = np.zeros(4)
		self.state[0:2] = self.initial_conditions
		self.state[2:4] = self.target_position

		# Normalise state
		self.state = self.state / np.array([2*np.pi, 2*np.pi, 2*np.pi, l1+l2])

	def initial_draw(self, canvas, initial_conditions):
		self.theta1 = initial_conditions[0]
		self.theta2 = initial_conditions[1]

		# Draw the first limb
		limb1_x1, limb1_y1 = get_limb_end(x_center, y_center, l1, self.theta1)
		self.limb1 = canvas.create_line(x_center, y_center, limb1_x1, limb1_y1, width=linewidth)

		# Draw the centre pivot
		cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1 = get_corners(x_center,y_center,10)
		self.cpivot = canvas.create_oval(cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1, fill="black")

		# Draw the second limb
		limb2_x1, limb2_y1 = get_limb_end(limb1_x1, limb1_y1, l2, self.theta2)
		self.limb2 = canvas.create_line(limb1_x1, limb1_y1, limb2_x1, limb2_y1, width=linewidth)

		# Draw the first bob
		bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(limb1_x1, limb1_y1, bob1_radius)
		self.bob1 = canvas.create_oval(bob1_x0, bob1_y0, bob1_x1, bob1_y1, fill="black")

		# Draw the second bob
		bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius)
		self.bob2 = canvas.create_oval(bob2_x0, bob2_y0, bob2_x1, bob2_y1, fill="#227C9D")

		# Draw text boxes
		self.text_box_1 = canvas.create_text(110,20,fill="black",font="Times 20 italic bold",
	                        text=f"Limb1 speed : ")
		self.text_box_2 = canvas.create_text(230,20,fill="black",font="Times 20 italic bold",
							text=f"0")
		self.text_box_3 = canvas.create_text(110,60,fill="black",font="Times 20 italic bold",
							text=f"Limb2 speed : ")
		self.text_box_4 = canvas.create_text(230,60,fill="black",font="Times 20 italic bold",
							text=f"0")

		# Draw hits text box
		self.text_box_rewards_1 = canvas.create_text(130,100,fill="black",font="Times 20 italic bold",
							text=f"Number of hits : ")
		self.text_box_rewards_2 = canvas.create_text(300,100,fill="black",font="Times 20 italic bold",
							text=f"{self.hits}")

		# Create first target
		self.create_target()

	# Creates a target circle in a random position on the screen
	# The target must be within the circle of radius l1+l2
	def create_target(self):
		random_angle = generate_random_angle()
		random_radius = np.random.uniform(0, l1+l2)
		x1, y1 = get_limb_end(window_width/2, window_height/2, random_radius, random_angle)
		x0, y0, x1, y1 = get_corners(x1, y1, target_radius)
		self.target = self.canvas.create_oval(x0, y0, x1, y1, fill="#C42021", outline="#C42021")
		self.canvas.tag_lower(self.target)
		self.target_position = np.array([random_angle, random_radius])

	# Move robot arm
	def move_arm(self, omega1, omega2):
		self.theta1 += omega1
		self.theta2 += omega2

		# Angles cannot exceed 2pi!
		if self.theta1 >= 2*np.pi:
			self.theta1 = self.theta1 - 2*np.pi
		if self.theta2 >= 2*np.pi:
			self.theta2 = self.theta2 - 2*np.pi

		# Angles cannot be less than 0
		if self.theta1 < 0:
			self.theta1 = self.theta1 + 2*np.pi
		if self.theta2 < 0:
			self.theta2 = self.theta2 + 2*np.pi

		limb1_x1, limb1_y1 = get_limb_end(x_center, y_center, l1, self.theta1)
		limb2_x1, limb2_y1 = get_limb_end(limb1_x1, limb1_y1, l2, self.theta2)
		bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(limb1_x1, limb1_y1, bob1_radius)
		bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius)

		self.canvas.coords(self.limb1, x_center, y_center, limb1_x1, limb1_y1)
		self.canvas.coords(self.limb2, limb1_x1, limb1_y1, limb2_x1, limb2_y1)
		self.canvas.coords(self.bob1, bob1_x0, bob1_y0, bob1_x1, bob1_y1)
		self.canvas.coords(self.bob2, bob2_x0, bob2_y0, bob2_x1, bob2_y1)
		self.canvas.itemconfig(self.text_box_2, text=f"{omega1}")
		self.canvas.itemconfig(self.text_box_4, text=f"{omega2}")
		self.canvas.itemconfig(self.text_box_rewards_2, text=f"{self.hits}")
		self.state[0:2] = np.array([self.theta1, self.theta2])
		# Normalise state
		self.state[0:2] = self.state[0:2] / np.array([2*np.pi, 2*np.pi])
		#time.sleep(0.001)

	def check_target(self):
		target_x0, target_y0, target_x1, target_y1 = self.canvas.coords(self.target)
		target_x_center = target_x0 + (target_x1 - target_x0)/2
		target_y_center = target_y0 + (target_y1 - target_y0)/2

		bob2_x0, bob2_y0, bob2_x1, bob2_y1 = self.canvas.coords(self.bob2)
		bob2_x_center = bob2_x0 + (bob2_x1 - bob2_x0)/2
		bob2_y_center = bob2_y0 + (bob2_y1 - bob2_y0)/2

		distance_between_centers = np.sqrt((target_x_center - bob2_x_center)**2 + (target_y_center - bob2_y_center)**2)
		max_distance = bob2_radius + target_radius

		if distance_between_centers < max_distance:
			# hit
			self.canvas.delete(self.target)
			self.create_target()
			self.state[2:4] = self.target_position
			# Normalise state
			self.state[2:4] = self.state[2:4] / np.array([2*np.pi, l1+l2])
			reward = 100
			self.hits += 1
			is_terminal = True
		else:
			#reward = 1/(distance_between_centers - max_distance)
			reward = -1
			is_terminal = False
		#print(reward)
		return reward, is_terminal

	def game_step(self, omega1, omega2):
		self.move_arm(omega1, omega2)
		reward, is_terminal = self.check_target()
		self.window.update()
		next_state = self.state
		return reward, next_state, is_terminal

#######################################################################################

class Agent():
	def __init__(self, game):
		self.game = game
		self.time_step = 0
		self.experiences = []
		self.create_neural_network()

	def create_neural_network(self):
		self.model = torch.nn.Sequential(
			torch.nn.Linear(layer1, layer2),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(layer2, layer3),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(layer3, layer4),
			torch.nn.Softmax(dim=0))

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
		'''
		try:

			checkpoint = torch.load('./statedict.pt')
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			#self.loss.load_state_dict(checkpoint['loss'])
			print(checkpoint['keyword'])
			print("Loaded network")
		except:
			print("No network to load")
			pass
		'''
		self.model.train()

	def uniform_random(self, q_values):
		return np.random.choice(np.arange(len(actions)))

	def discount_rewards(self, rewards):
		discounted_return = torch.pow(gamma, torch.arange(len(rewards)).float())*rewards
		discounted_return /= discounted_return.max()
		return discounted_return

	def loss_function(self, probabilities, discounted_return):
		loss = -1*torch.sum(discounted_return*torch.log(probabilities))
		return loss

	def play(self):
		is_terminal = False
		self.experiences = np.zeros(10)
		while is_terminal == False:
			# Choose action
			state1 = self.game.state
			#print(f"State: {state1}")
			experience = np.zeros(10)
			experience[0:4] = state1
			probabilities = self.model(torch.from_numpy(state1).float())
			action_index = np.random.choice(np.arange(len(actions)), p=probabilities.data.numpy())
			#action_index = np.random.choice(np.arange(len(actions)))
			action = actions[action_index]
			experience[4] = action_index
			omega1 = action[0]
			omega2 = action[1]
			# Observe next state and reward
			reward, state2, is_terminal = self.game.game_step(omega1, omega2)
			experience[5] = reward
			experience[6:] = state2
			if len(self.experiences) >= maximum_batch_size:
				self.experiences = np.delete(self.experiences, 1, 0)
			self.experiences = np.vstack((self.experiences,experience))
			self.time_step += 1
		
		# Add episode length to hit_time_steps
		hit_time_steps.append(self.time_step)

		# Remove zero row from experiences
		self.experiences = self.experiences[1:]
		# Change experiences to torch Tensor
		self.experiences = torch.from_numpy(self.experiences).float()

		reward_batch = self.experiences[:,5]
		discounted_return = self.discount_rewards(reward_batch)
		state_batch = self.experiences[:,0:4]
		action_index_batch = self.experiences[:,4]
		predictions_batch = self.model(state_batch)
		# Only include probabilities of the actions taken
		predictions_batch_actions_taken = predictions_batch.gather(dim=1,index=action_index_batch.long().view(-1,1)).squeeze()
		loss = self.loss_function(predictions_batch_actions_taken, discounted_return)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Save learned weights and biases
		try:
			torch.save({
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'loss': loss,
				'keyword': 'thisthing2'},
				'statedict.pt')
			print("Saved!")	
		except:
			pass

#######################################################################################

if __name__ == '__main__':
	game = PendulumGame()
	agent = Agent(game)
	for session in range(training_sessions):
		agent.play()
		print(len(agent.experiences))
	print(hit_time_steps)
	print(np.average(np.array(hit_time_steps)))
	game.window.mainloop()