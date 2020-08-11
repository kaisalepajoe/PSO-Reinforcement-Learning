# This is the double pendulum environment

import numpy as np 
import tkinter as tk 
import time
from scipy.integrate import odeint
import torch

#####################################################

# Used resources

# Inspiration for how to solve Euler Lagrange https://scipython.com/blog/the-double-pendulum/
# Tutorial for scipy's odeint https://www.youtube.com/watch?v=VV3BnroVjZo

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
target_radius = 29
m1 = 4
m2 = 4
g = 9.81
theta_dot_max = 1.9

# Set neural network parameters
actions = np.array([-0.1, -0.01, -0.005, 0, 0.005, 0.01, 0.1])
layer1 = 6
layer2 = 150
layer3 = 100
layer4 = len(actions)
learning_rate = 0.001
gamma = 0.9
temperature = 10

#######################################################################################

# Create window
def create_window():
	window = tk.Tk()
	canvas = tk.Canvas(window, width=window_width, height=window_height, bg="#FFFFFC")
	canvas.grid() # tell tkinter where to place the canvas
	window.title("Double Pendulum Environment")
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
	z1 = 0
	z2 = 0

	return [theta1, z1, theta2, z2]

def generate_random_angle():
	'''
	Returns a random angle.
	'''
	random_angle = np.random.random()*2*np.pi
	return random_angle

# Returns derivatives for scipy's odeint function
def derivatives(y, t, l1, l2, m1, m2, omega):
	# testing omega
	# swapping all z1 for z1 + omega with constant omega.
	# Expecting a constant acceleration, leading to the pendulum spinning around counterclockwise
	theta1, z1, theta2, z2 = y

	dtheta1_dt = z1 + omega
	dtheta2_dt = z2

	dz1_dt = (m2*g*np.sin(theta2)*np.cos(theta1-theta2) - m2*np.sin(theta1-theta2)*(l1*(z1+omega)**2*np.cos(theta1-theta2)+l2*z2**2) - (m1+m2)*g*np.sin(theta1))/\
		(l1*(m1+m2*np.sin(theta1-theta2)**2))
	dz2_dt = ((m1+m2)*(l1*(z1+omega)**2*np.sin(theta1-theta2)-g*np.sin(theta2)+g*np.sin(theta1)*np.cos(theta1-theta2))+m2*l2*z2**2*np.sin(theta1-theta2)*np.cos(theta1-theta2))/\
		(l2*(m1+m2*np.sin(theta1-theta2)**2))

	return dtheta1_dt, dz1_dt, dtheta2_dt, dz2_dt

#######################################################################################

class PendulumGame():
	def __init__(self):

		self.window, self.canvas = create_window()

		self.initial_conditions = generate_initial_conditions()
		self.hits = 0

		self.initial_draw(self.canvas, self.initial_conditions)

		self.state = np.inf*np.ones(6)
		self.state[0:4] = self.initial_conditions
		self.state[4:6] = self.target_position

	def initial_draw(self, canvas, initial_conditions):
		self.theta1 = initial_conditions[0]
		self.theta2 = initial_conditions[2]

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
		self.bob1 = canvas.create_oval(bob1_x0, bob1_y0, bob1_x1, bob1_y1, fill="#44AFD6")

		# Draw the second bob
		bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius)
		self.bob2 = canvas.create_oval(bob2_x0, bob2_y0, bob2_x1, bob2_y1, fill="#227C9D")

		# Draw text boxes
		self.text_box_1 = canvas.create_text(80,20,fill="black",font="Times 20 italic bold",
	                        text=f"Omega : ")
		self.text_box_2 = canvas.create_text(160,20,fill="black",font="Times 20 italic bold",
							text=f"0")

		# Draw hits text box
		self.text_box_rewards_1 = canvas.create_text(144,60,fill="black",font="Times 20 italic bold",
							text=f"Number of hits : ")
		self.text_box_rewards_2 = canvas.create_text(300,60,fill="black",font="Times 20 italic bold",
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
		self.target_position = np.array([x1, y1])

	# Move pendulum
	def move_pendulum(self, time_step, omega):

		time_array = np.arange(time_step,time_step+2)
		assert len(time_array) == 2
		assert time_array[0] == time_step
		y = odeint(derivatives, self.initial_conditions, time_array, args=(l1, l2, m1, m2, omega))
		self.theta1 = y[0][0]
		self.theta2 = y[0][2]
		self.initial_conditions = y[1]

		limb1_x1, limb1_y1 = get_limb_end(x_center, y_center, l1, self.theta1)
		limb2_x1, limb2_y1 = get_limb_end(limb1_x1, limb1_y1, l2, self.theta2)
		bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(limb1_x1, limb1_y1, bob1_radius)
		bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius)

		self.canvas.coords(self.limb1, x_center, y_center, limb1_x1, limb1_y1)
		self.canvas.coords(self.limb2, limb1_x1, limb1_y1, limb2_x1, limb2_y1)
		self.canvas.coords(self.bob1, bob1_x0, bob1_y0, bob1_x1, bob1_y1)
		self.canvas.coords(self.bob2, bob2_x0, bob2_y0, bob2_x1, bob2_y1)
		self.canvas.itemconfig(self.text_box_2, text=f"{omega}")
		self.canvas.itemconfig(self.text_box_rewards_2, text=f"{self.hits}")
		self.state[0:4] = np.array(y[0])
		time.sleep(0.05)

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
			self.state[4:6] = self.target_position
			reward = 500
			self.hits += 1
		else:
			reward = -1

		# Also check for maximum speed
		theta1_dot = self.state[1]
		theta2_dot = self.state[3]
		combined_theta_dot = np.sqrt((theta1_dot)**2 + (theta2_dot)**2)

		if combined_theta_dot > theta_dot_max:
			reward -= (combined_theta_dot - theta_dot_max)*50
			print(f"Too fast! {reward}")

		return reward

	def game_step(self, time_step, omega):
		self.move_pendulum(time_step, omega)
		reward = self.check_target()
		self.window.update()
		next_state = self.state
		return reward, next_state

#######################################################################################

class Agent():
	def __init__(self, game):
		self.game = game
		self.state = game.state
		self.time_step = 0
		self.create_neural_network()

	def create_neural_network(self):
		self.model = torch.nn.Sequential(
			torch.nn.Linear(layer1, layer2),
			torch.nn.ReLU(),
			torch.nn.Linear(layer2, layer3),
			torch.nn.ReLU(),
			torch.nn.Linear(layer3, layer4),
			torch.nn.ReLU())

		self.loss_fn = torch.nn.MSELoss() # mean squared error
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

		try:
			checkpoint = torch.load('./statedict2.pt')
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.loss = checkpoint['loss']
			print("Loaded network")
		except:
			pass
		self.model.train()

	def get_q_values_t(self, state, no_grad=False):
		state_t = torch.from_numpy(state).float()
		if no_grad == True:
			with torch.no_grad():
				q_values_t = self.model(state_t)
		else:
			q_values_t = self.model(state_t)
		return q_values_t

	def softmax(self, q_values_t):
		q_values = q_values_t.detach().numpy()
		max_q_value = np.max(q_values)
		numerator = np.exp((q_values-max_q_value)/temperature)
		denominator = np.sum(np.exp((q_values-max_q_value)/temperature))
		probabilities = numerator/denominator
		omega_index = np.random.choice(np.arange(0,len(actions)),p=probabilities)
		return omega_index

	def play(self):
		while True:
			q_values_t = self.get_q_values_t(self.state)
			omega_index = self.softmax(q_values_t)
			omega = actions[omega_index]
			reward, next_state = self.game.game_step(self.time_step, omega)
			if reward >= 449:
				print("Saved!")
				# Save learned weights and biases
				torch.save({
	        		'model_state_dict': self.model.state_dict(),
	        		'optimizer_state_dict': self.optimizer.state_dict(),
	        		'loss': self.loss}, 'statedict2.pt')
			self.update_model(reward, next_state, q_values_t, omega_index)
			self.state = next_state
			self.time_step += 1

	def update_model(self, reward, next_state, q_values_t, omega_index):
		next_q_values_t = self.get_q_values_t(next_state, no_grad=True)
		next_q_values = next_q_values_t.numpy()
		max_next_q_value = np.max(next_q_values)
		target_q_value = reward + gamma*max_next_q_value

		updated_q_values_t = q_values_t.clone()
		updated_q_values_t[omega_index] = target_q_value

		self.loss = self.loss_fn(q_values_t, updated_q_values_t)
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

#######################################################################################

if __name__ == '__main__':
	game = PendulumGame()
	agent = Agent(game)
	agent.play()
	game.window.mainloop()