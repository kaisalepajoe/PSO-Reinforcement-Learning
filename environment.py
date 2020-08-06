# This is the double pendulum environment

import numpy as np 
import simpy
import tkinter as tk 
import time
from scipy.integrate import odeint

env = simpy.Environment()

#####################################################

# Used resources

# Inspiration for how to solve Euler Lagrange https://scipython.com/blog/the-double-pendulum/
# Tutorial for scipy's odeint https://www.youtube.com/watch?v=VV3BnroVjZo
# simpy tutorial https://realpython.com/simpy-simulating-with-python/

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
target_radius = 5
m1 = 4
m2 = 4
g = 9.81

total_time_steps = 100_000

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

# Get the x and y coordinate of the end of the limb
def get_limb_end(x0, y0, length, angle):
	x1 = x0 + length*np.sin(angle)
	y1 = y0 + length*np.cos(angle)
	return x1, y1

def generate_random_angle():
	'''
	Returns a random angle between 0 and 2pi. No parameters.
	'''
	return np.random.random()*2*np.pi

# Draw all shapes with initial positions
def initial_draw(canvas, initial_conditions):
	theta1 = initial_conditions[0]
	theta2 = initial_conditions[2]

	# Draw the first limb
	limb1_x1, limb1_y1 = get_limb_end(x_center, y_center, l1, theta1)
	limb1 = canvas.create_line(x_center, y_center, limb1_x1, limb1_y1, width=linewidth)

	# Draw the centre pivot
	cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1 = get_corners(x_center,y_center,10)
	cpivot = canvas.create_oval(cpivot_x0, cpivot_y0, cpivot_x1, cpivot_y1, fill="black")

	# Draw the second limb
	limb2_x1, limb2_y1 = get_limb_end(limb1_x1, limb1_y1, l2, theta2)
	limb2 = canvas.create_line(limb1_x1, limb1_y1, limb2_x1, limb2_y1, width=linewidth)

	# Draw the first bob
	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(limb1_x1, limb1_y1, bob1_radius)
	bob1 = canvas.create_oval(bob1_x0, bob1_y0, bob1_x1, bob1_y1, fill="#44AFD6")

	# Draw the second bob
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius)
	bob2 = canvas.create_oval(bob2_x0, bob2_y0, bob2_x1, bob2_y1, fill="#227C9D")

	# Draw text box
	text_box = canvas.create_text(80,20,fill="black",font="Times 20 italic bold",
                        text=f"Omega : {0}")

	elements = {
		"theta1":theta1,
		"theta2":theta2,
		"limb1":limb1,
		"limb2":limb2,
		"bob1":bob1,
		"bob2":bob2,
		"cpivot":cpivot,
		"text_box":text_box
	}
	return elements

# Returns derivatives for scipy's odeint function
def derivatives(y, t, l1, l2, m1, m2, omega):
	# testing omega
	# swapping all z1 for z1 + omega with constant omega.
	# Expecting a constant acceleration, leading to the pendulum spinning around counterclockwise
	theta1, z1, theta2, z2 = y

	dtheta1_dt = (z1 + omega)
	dtheta2_dt = z2

	dz1_dt = (m2*g*np.sin(theta2)*np.cos(theta1-theta2) - m2*np.sin(theta1-theta2)*(l1*(z1+omega)**2*np.cos(theta1-theta2)+l2*z2**2) - (m1+m2)*g*np.sin(theta1))/\
		(l1*(m1+m2*np.sin(theta1-theta2)**2))
	dz2_dt = ((m1+m2)*(l1*(z1+omega)**2*np.sin(theta1-theta2)-g*np.sin(theta2)+g*np.sin(theta1)*np.cos(theta1-theta2))+m2*l2*z2**2*np.sin(theta1-theta2)*np.cos(theta1-theta2))/\
		(l2*(m1+m2*np.sin(theta1-theta2)**2))

	return dtheta1_dt, dz1_dt, dtheta2_dt, dz2_dt

def generate_initial_conditions():
	theta1 = generate_random_angle()
	theta2 = generate_random_angle()
	z1 = 0
	z2 = 0

	return [theta1, z1, theta2, z2]

# Creates a target circle in a random position on the screen
# The target must be within the circle of radius l1+l2
def create_target(canvas):
	random_angle = np.random.random()*2*np.pi
	random_radius = np.random.uniform(0, l1+l2)
	x1, y1 = get_limb_end(window_width/2, window_height/2, random_radius, random_angle)
	x0, y0, x1, y1 = get_corners(x1, y1, target_radius)
	target = canvas.create_oval(x0, y0, x1, y1, fill="#C42021", outline="#C42021")
	return target

# Move pendulum
def move_pendulum(elements, time_step, initial_conditions, omega):

	time_array = np.arange(time_step,time_step+2)
	assert len(time_array) == 2
	assert time_array[0] == time_step
	y = odeint(derivatives, initial_conditions, time_array, args=(l1, l2, m1, m2, omega))
	theta1 = y[-1][0]
	theta2 = y[-1][2]
	initial_conditions = y[1]

	limb1_x1, limb1_y1 = get_limb_end(x_center, y_center, l1, theta1)
	limb2_x1, limb2_y1 = get_limb_end(limb1_x1, limb1_y1, l2, theta2)
	bob1_x0, bob1_y0, bob1_x1, bob1_y1 = get_corners(limb1_x1, limb1_y1, bob1_radius)
	bob2_x0, bob2_y0, bob2_x1, bob2_y1 = get_corners(limb2_x1, limb2_y1, bob2_radius)

	canvas.coords(elements["limb1"], x_center, y_center, limb1_x1, limb1_y1)
	canvas.coords(elements["limb2"], limb1_x1, limb1_y1, limb2_x1, limb2_y1)
	canvas.coords(elements["bob1"], bob1_x0, bob1_y0, bob1_x1, bob1_y1)
	canvas.coords(elements["bob2"], bob2_x0, bob2_y0, bob2_x1, bob2_y1)
	canvas.itemconfig(elements["text_box"], text=f"Omega : {omega}")

	time.sleep(0.05)

	return theta1, theta2, initial_conditions

def run_simulation(total_time_steps, elements, initial_conditions):
	time_step = 0
	omega = 0
	while True:
		# try checking for button presses

		theta1, theta2, initial_conditions = move_pendulum(elements, time_step, initial_conditions, omega)
		elements["theta1"] = theta1
		elements["theta2"] = theta2
		window.update()
		time_step += 1
	window.mainloop() # draw the window
'''
environment stuff that may be useful later
def bob(env):
	while True:
		print(f"Moving pendulum {env.now}")
		moving_duration = 5
		yield env.timeout(moving_duration)

		print(f"Pendulum stopped {env.now}")
		stop_duration = 3
		yield env.timeout(stop_duration)

#env.process(bob(env))
'''

if __name__ == '__main__':
	finish_times = []
	env = simpy.Environment()
	window, canvas = create_window()
	initial_conditions = generate_initial_conditions()
	target = create_target(canvas)
	elements = initial_draw(canvas, initial_conditions)
	run_simulation(total_time_steps, elements, initial_conditions)
	env.run()