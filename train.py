# Particle Swarm Optimization
# This script finds the global MINIMUM of the
# selected function.

# This algorithm is inspired by PSO(0) from the book 
# "Particle Swarm Optimization" by Maurice Clerc.

###################################################################

# Import required modules
import numpy as np 
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pkgutil
import io
import sys
from tqdm import tqdm
from environment import RobotArmGame
import torch

###################################################################

# Helper functions

def determine_error(found_value, minimum_value=0):
	'''
	Returns the difference between two values.
	Parameters
	----------
	found_value : number
	A value found in the experiment.
	minimum_value : number
	The known value.
	Returns
	-------
	error : number
	The absolute difference between the found value and known value.
	'''
	error = abs(found_value - minimum_value)
	return error

def n_evaluations(N, time_steps, repetitions):
	'''
	Returns the number of evaluations of an experiment.
	Returns the number of evaluations for a given number of particles,
	time_steps, and repetitions.
	If given an array, does calculations on rows and returns the number of
	evaluations as an array.
	Parameters
	----------
	N : number or np.ndarray
	The number of particles in the swarm.
	time_steps : number or np.ndarray
	The number of time steps in each swarm evolution.
	
	repetitions : number or np.ndarray
	The number of times swarm evolutions are repeated.
	Returns
	-------
	n_evaluations : number or np.ndarray
	'''
	n_evaluations = N*time_steps*repetitions + repetitions*N
	if type(N) == np.ndarray or type(time_steps) == np.ndarray\
		or type(repetitions) == np.ndarray:
		return (n_evaluations).astype(int)
	else:
		return math.ceil(n_evaluations)

###################################################################

class Experiment:
	'''
	This class sets the parameters from the constants and fn_info dictionaries
	as attributes of the Class. The class is inherited by both Particles and Swarms.
	The main method of this script is 'run' from the Experiment class.
	Create an experiment object: experiment = ob.Experiment()
	Run the PSO algorithm: experiment.run()
	'''
	def __init__(self, constants=None, fn_info=None):
		'''
		This function sets the parameters from the dictionaries 'constants' and 'fn_info'
		'constants' contains: phi, k, N, time_steps, repetitions
		'fn_info' contains: fn_name, optimal_f, dim, xmin, xmax, param_is_integer,
		special_constraints, constraints_function, constraints_extra_arguments,
		show_animation, disable_progress_bar, get_parameters_from.
		Parameters
		----------
		constants : dict
		The dictionary containing the following hyperparameters of the PSO algorithm.
			phi : number
			Phi sets the two confidence parameters c1 and cmax as described in the PSO book.
			
			k : int
			The number of informants for each particle.
			
			N : int
			The number of particles in the swarm.
			
			time_steps : int
			The number of time steps in each evolution of the swarm.
			
			repetitions : int
			The number of times a swarm is evolved before taking the average best position and value.
		
		fn_info : dict
		The dictionary containing the following information about the function to be evaluated.
			
			fn_name : str
			The name of the function to be evaluated by the PSO. For example, 'Rosenbrock'.
			
			optimal_f : number
			The value of the best position, used to calculate the error. For Rosenbrock, this is 0.
			
			dim : int
			The number of dimensions of the problem.
			
			xmin : list
			A list of minimum values for the search space. Each element corresponds to a dimension.
			
			xmax : list
			A list of maximum values for the search space. Each element corresponds to a dimension.
			
			param_is_integer : list of bools
			A list of booleans for each dimension. True means that the parameter can only be an integer.
			
			special_constraints : bool
			True if there are any special constraints. False if the only constraints are a rectangular
			search space given by xmin and xmax.
			
			constraints_function : str
			The name of the function that applies any special constraints to the parameters visited by
			the particles. Used if special_constraints is set to True.
			
			constraints_extra_arguments : list
			A list of arguments passed to the constraints function if special_constraints is set to True.
			The first element of the list must be a boolean that indicates if initial positions must be
			generated. This boolean should initially be set to True, so the constraint function knows to 
			generate initial positions, not next positions.
			show_animation : bool
			An animation of the swarm is shown at the end of the Experiment.run function if this argument
			is set to True. 
			
			disable_progress_bar : bool
			No progress bar is printed if this argument is set to True.
			
			get_parameters_from : str
			This string is either "g-values" or "average p-values". Used in the get_parameters method of
			the Swarm class. See the doc of this method for more information.
			If not sure, set to "g-values".
		Returns
		-------
		Nothing
		'''
		if np.all(constants == None):
			constants = read_dictionary_from_file('optimal_constants.txt')
		if np.all(fn_info == None):
			fn_info = read_dictionary_from_file('fn_info.txt')

		if type(constants) == dict and type(fn_info) == dict:
			self.N = constants["N"]
			self.time_steps = constants["time_steps"]
			self.repetitions = constants["repetitions"]
			self.fn_name = fn_info["fn_name"]
			self.optimal_f = fn_info["optimal_f"]
			self.dim = fn_info["dim"]
			self.k = constants["k"]
			self.phi = constants["phi"]
			self.xmin = np.array(fn_info["xmin"])
			self.xmax = np.array(fn_info["xmax"])
			self.param_is_integer = np.array(fn_info["param_is_integer"])
			self.show_animation = fn_info["show_animation"]
			self.special_constraints = fn_info["special_constraints"]
			self.constraints_function = fn_info["constraints_function"]
			self.constraints_extra_arguments = fn_info["constraints_extra_arguments"]
			self.disable_progress_bar = fn_info["disable_progress_bar"]
			self.disable_printing = fn_info["disable_printing"]
			self.get_parameters_from = fn_info["get_parameters_from"]

			# Calculate maximum velocity
			self.vmax = np.absolute(self.xmax - self.xmin)/2

			# Calculate confidence parameters using phi
			self.c1 = 1/(self.phi-1+np.sqrt(self.phi**2-2*self.phi))
			self.cmax = self.c1*self.phi

		else:
			raise TypeError(f"Invalid types {type(constants)} and {type(fn_info)} for constants and fn_info.")

	# Return dictionary of current constants if argument 'dictionary' is not given
	# Update current constants if 'dictionary' is given and return the given dictionary
	def constants(self, constants_dictionary=None):
		'''
		Returns dictionary of current constants if argument 'constants_dictionary' is not given.
		If the argument 'constants_dictionary' is given, then sets the constants to the values
		found in 'constants_dictionary', and returns 'constants_dictionary'.
		Parameters
		----------
		constants_dictionary : dict
.
		Returns
		-------
		constants : dict
		The constants set for this Experiment object.
		'''
		if constants_dictionary == None:
			constants = {'phi': self.phi, 'N': self.N, 'k': self.k, 
				'time_steps': self.time_steps, 'repetitions': self.repetitions}
		elif type(constants_dictionary) == dict:
			constants = constants_dictionary
			self.phi = constants["phi"]
			self.N = constants["N"]
			self.k = constants["k"]
			self.time_steps = constants["time_steps"]
			self.repetitions = constants["repetitions"]
		else:
			raise TypeError(f"Invalid type {type(constants_dictionary)} for dictionary")

		return constants

	def fn_info(self, fn_info_dictionary=None):
		'''
		Returns dictionary of current fn_info if argument 'fn_info_dictionary' is not given.
		If the dictionary is given, then sets the constants to the values
		found in the dictionary, and returns the same dictionary.
		Parameters
		----------
		fn_info_dictionary : dict
		The dictionary containing the constants phi, k, N, time_steps, repetitions.
		Returns
		-------
		fn_info : dict
		The function info set for this experiment object.
		'''
		if fn_info_dictionary == None:
			fn_info = {"fn_name":self.fn_name, "optimal_f":self.optimal_f, "dim":self.dim,
				"xmin":self.xmin.tolist(), "xmax":self.xmax.tolist(), 
				"param_is_integer":self.param_is_integer.tolist(),
				"special_constraints":self.special_constraints,
				"constraints_function":self.constraints_function,
				"constraints_extra_arguments":self.constraints_extra_arguments,
				"show_animation":self.show_animation,
				"disable_progress_bar":self.disable_progress_bar,
				"disable_printing":self.disable_printing,
				"get_parameters_from": self.get_parameters_from}
			return fn_info
		elif type(fn_info_dictionary) == dict:
			fn_info = fn_info_dictionary
			return fn_info
		else:
			raise TypeError(f"Invalid type {type(fn_info_dictionary)} for dictionary")

	def n_evaluations(self):
		'''
		Returns the number of evaluations.
		This function uses the n_evaluations helper function to calculate the
		number of evaluations, but sets the parameters N, time_steps, repetitions
		automatically. It is useful to get the number of evaluations of an experiment
		object quickly.
		Parameters
		----------
		None
		Returns
		-------
		n_evaluations : number
		The number of evaluations of an Experiment object, given the current constants
		configuration.
		'''
		return n_evaluations(self.N, self.time_steps, self.repetitions)

	def run(self, allowed_n_evaluations=None):
		'''
		Runs the experiment from beginning to end. 
		Returns the best found position, best value and error,
		and also assigns these to the Experiment object as attributes.
		If show_animation is set to True, the swarm will be animated.
		To see the animation again, use the command experiment.swarm.animate_swarm()
		Parameters
		----------
		allowed_n_evaluations : number
		The maximum number of evaluations allowed for this run. The true number of 
		evaluations bight be a bit higher or lower than this value.
		Returns
		-------
		best_position : np.ndarray
		An array of the best performing parameters found.
		best_f : float
		The value of the function, for example 'Rosenbrock', at the best_position.
		error : float
		The difference between best_f and the known best value optimal_f from fn_info.
		'''

		# Check if user has given a new number of evaluations.
		if allowed_n_evaluations == None:
			allowed_n_evaluations = n_evaluations(self.N, self.time_steps, self.repetitions)
		else:
			if allowed_n_evaluations <= 2*self.N*self.repetitions:
				raise ValueError(f"Number of evaluations must be greater than 2Nr. In this case >= {2*self.N*self.repetitions}")
		
		# Recalculate the time_steps to achieve this maximum number of evaluations.
		self.time_steps = math.ceil((allowed_n_evaluations - self.repetitions*self.N)/(self.repetitions*self.N))
		print("Running algorithm...")

		constants = self.constants()
		fn_info = self.fn_info()

		# Create swarm and evolve the swarm for the required number of repetitions.
		self.swarm = Swarm(constants, fn_info)
		self.swarm.distribute_swarm()
		self.swarm.run_algorithm()
		true_n_evaluations = n_evaluations(self.swarm.N, self.swarm.time_steps, self.swarm.repetitions)

		self.best_position = self.swarm.best_position
		self.best_f = self.swarm.best_f
		self.error = self.swarm.error

		if self.disable_printing == False:
			print(f"{true_n_evaluations} evaluations made.")
			print(f"The best position is {repr(self.best_position.tolist())}.")
			print(f"The value at this position is {self.best_f}")
			print(f"Error in value: {self.error}")

		if self.show_animation == False:
			pass
		else:
			self.swarm.animate_swarm()

		return self.best_position, self.best_f, self.error

###################################################################

class Swarm(Experiment):
	'''
	The Swarm class inherits the __init__ function from the Experiment class.
	This class creates Particle instances and evolves them through time to get a final
	best position, value, and error.
	'''
	def random_initial_positions(self):
		'''
		Returns an array of random initial positions.
		Returns an array of random initial positions for each particle
		in a swarm. Each position is within the search space given by xmin, xmax,
		and any special constraints in the fn_info dictionary.
		Parameters
		----------
		None
		Returns
		-------
		initial_positions : np.ndarray
		An array of random initial positions that are within the required search area.
		The array has dimensions (number of particles, number of dimensions of the problem).
		'''
		initial_positions = np.inf*np.ones((self.N, self.dim))
		# Check if there are any special constraints
		if self.special_constraints == False:
			# Create array of initial positions
			# taking into account that some parameters must be integers
			for d in range(self.dim):
				if self.param_is_integer[d] == True:
					initial_positions[:,d] = np.random.randint(self.xmin[d], self.xmax[d], self.N)
				elif self.param_is_integer[d] == False:
					initial_positions[:,d] = np.random.uniform(self.xmin[d], self.xmax[d], self.N)
			# Note that these positions are all of type np.float64 even though randint is called
		else:
			for particle in range(self.N):
				initial_positions[particle] = eval(self.constraints_function)(None, self.constraints_extra_arguments)

		return initial_positions

	def random_initial_velocities(self):
		'''
		Returns an array of random initial velocities for the Particles.
		Parameters 
		----------
		None
		Returns
		-------
		initial_velocities : np.ndarray
		An array of random initial velocities that are within [-vmax, vmax).
		The array has dimensions (number of particles, number of dimensions of the problem).
		'''
		initial_velocities = np.random.uniform(-self.vmax, self.vmax, (self.N, self.dim))

		return initial_velocities

	def create_particles(self, initial_positions, initial_velocities):
		'''
		Creates a list of Particle objects for the swarm.
		Parameters
		----------
		initial_positions : array
		An array of initial positions for all N particles
		with shape (number of particles, number of dimensions)
		initial_velocities : array
		An array of initial velocities for all N particles
		with shape (number of particles, number of dimensions)
		Returns
		-------
		None
		'''

		# Create array of initial p-values by evaluating initial positions
		p_values = np.inf*np.ones((self.N, self.dim+1))
		for i, pos in enumerate(initial_positions):
			p_values[i,0:self.dim] = pos		
			if self.special_constraints == True:
				value = eval(self.fn_name)(pos)
			else:
				value = eval(self.fn_name)(pos)
			p_values[i,self.dim] = value

		constants = self.constants()
		fn_info = self.fn_info()

		# Create list of particle objects
		self.particles = []
		for i in range(self.N):
			pos = initial_positions[i]
			vel = initial_velocities[i]
			p = p_values[i]
			particle = Particle(constants, fn_info)
			particle.set_initial_state(pos, vel, p)
			self.particles.append(particle)

	def random_informants(self):
		'''
		Chooses k informants randomly for each Particle of the Swarm.
		Sets a list of these informant Particles as attributes for each Particle.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''
		for particle in self.particles:
			particle.informants = np.random.choice(self.particles, particle.k)

	def distribute_swarm(self):
		'''
		Distributes Particles in the search space and chooses informants for each particle.
		Also initializes an array of positions for animating the Swarm.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''

		# Create array of initial positions and velocities
		initial_positions = self.random_initial_positions()
		initial_velocities = self.random_initial_velocities()

		self.create_particles(initial_positions, initial_velocities)

		# Initiate k informants randomly
		self.random_informants()

		# Initialise array of positions for animation
		self.positions = np.inf*np.ones((self.time_steps, self.N, self.dim))
		self.positions[0,:,:] = initial_positions

	def evolve(self):
		'''
		Updates positions of Particles for all time steps. Populates the positions array
		for animating the Swarm. Also shows a progress bar.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''

		# Evolve swarm for all time steps
		for time_step in tqdm(range(self.time_steps),
			desc=f"Repetition {self.current_repetition}/{self.repetitions}: Evolving swarm",
			disable=self.disable_progress_bar):
			for i, particle in enumerate(self.particles):
				particle.step()
				# Update positions for animation
				self.positions[time_step,i,:] = particle.pos
			# Select informants for next time step
			self.random_informants()


	def get_parameters(self):
		'''
		Returns optimal parameters and lowest value found.
		If get_parameters_from is set to 'g-values' in the fn_info dictionary,
		then the optimal parameters are chosen from the global values. The g-values
		of all Particles in the Swarm are inspected and the lowest value is chosen.
		If get_parameters_from is set to 'average p-values', then the
		optimal parameters are chosen from the best visited positions of each particle.
		The p-values of all Particles in the Swarm are inspected, and the average of 
		positions and values are returned.
		Parameters
		----------
		None
		Returns
		-------
		result : np.ndarray
		An array containing the best found parameter for each dimension and the value
		of the function with these parameters. The value is the last element of the array.
		'''
		if self.get_parameters_from == "g-values":
			final_g = np.inf*np.ones((self.N, self.dim+1))
			for i,particle in enumerate(self.particles):
				final_g[i,:] = particle.g
			optimal_i = np.argmin(final_g[:,self.dim])
			result = final_g[optimal_i]
		if self.get_parameters_from == "average p-values":
			final_p = np.inf*np.ones((self.N, self.dim+1))
			for i,particle in enumerate(self.particles):
				final_p[i,:] = particle.p 
			result = np.average(final_p, axis=0)
		return result

	def run_algorithm(self):
		'''
		Evolves the swarm through time for the required number of repetitions.
		Assigns the best found position, value, and error to the Swarm, and
		returns these as a tuple.
		Parameters
		---------
		None
		Returns
		-------
		(best_position, best_f, error)
		best_position : np.ndarray
		An array of the best parameters found.
		best_f : float
		The value of the function at this position.
		error : float
		The difference between the best_f and optimal_f given in the fn_info dictionary.
		'''
		results = np.inf*np.ones((self.repetitions, self.dim+1))
		# all_positions contains all the visited positions for each repetition
		# all_positions is used to create an animation of the swarm
		self.all_positions = np.inf*np.ones((self.repetitions, self.time_steps, self.N, self.dim))

		for r in range(self.repetitions):
			self.current_repetition = r+1
			self.distribute_swarm()
			self.evolve()
			result = self.get_parameters()
			results[r] = result
			self.all_positions[r] = self.positions

		self.best_value_index = np.argmin(results[:,self.dim])

		self.best_position = results[self.best_value_index][0:self.dim]
		self.best_f = results[self.best_value_index][self.dim]
		self.error = determine_error(self.best_f, self.optimal_f)

		return self.best_position, self.best_f, self.error


	def animate_swarm(self):
		'''
		Plots an animation of the best repetition of evolving the swarm.
		Only the first 2 dimensions are plotted for a higher-dimensional problem.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''

		# Plot initial positions of particles
		fig, ax = plt.subplots()
		ax.set_xlim(self.xmin[0], self.xmax[0])
		ax.set_ylim(self.xmin[1], self.xmax[1])
		scat = ax.scatter(self.all_positions[self.best_value_index,0,:,0], 
			self.all_positions[self.best_value_index,0,:,1], color="Black", s=2)

		# Create animation
		interval = 200_000 / (self.N * self.time_steps * self.repetitions)
		self.animation = FuncAnimation(fig, func=self.update_frames, interval=interval, 
			fargs=[scat, self.all_positions, self.best_value_index])
		plt.show()

	def update_frames(self, j, *fargs):
		'''
		Updates the frames of the animation.
		Parameters
		----------
		j : int
		Frame number.
		*fargs contains scat, all_positions, and best_value_index
		scat : scatter plot
		all_positions : np.ndarray
		An array of all positions visited by the Swarm.
		best_value_index : int
		The index indicating the best performing repetition.
		Returns
		-------
		None
		'''
		scat, all_positions, best_value_index = fargs
		try:
			scat.set_offsets(all_positions[best_value_index,j,:,0:2])
		except:
			print("Animation finished.")
			self.animation.event_source.stop()

###################################################################

# Particle objects are created within the swarm class methods. 
class Particle(Experiment):
	'''
	Particle instances are created within the Swarm class methods. Particles
	inherit constants and function info from the Experiment class.
	'''

	def set_initial_state(self, pos, vel, p):
		'''
		Initializes a particle with the assigned initial position, initial velocity,
		p-value, g-value and an empty list of informants.
		
		Parameters
		----------
		pos : array
		An array containing an initial position for each dimension.
		vel : array
		An array containing an initial velocity for each dimension.
		p : array
		An array containing the best found position and value that the particle
		has visited. The initial state has p equal to the initial position and its
		value as the particle has not visited any other positions yet.
		Returns
		-------
		None
		'''
		self.pos = pos
		self.vel = vel
		# Set initial best found value by particle
		# format: np array of shape (1, 3) - x, y, value
		self.p = p

		# Best found position and value by informants or itself
		# format: np array of shape (1, 3) - x, y, value
		self.g = p

		# Empty list of informants
		self.informants = []

	def communicate(self):
		'''
		Receives g-values from informants and updates the Particle's g-value accordingly.
		If the best received g-value is smaller than the Particle's g-value, then the
		particles g-value is set to the received g-value.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''

		# Receive best positions with values from informants
		received = np.zeros((self.k, self.dim+1))
		for i, informant in enumerate(self.informants):
			received[i, :] = informant.g
		# Find best g from communicated values
		i = np.argmin(received[:,self.dim])
		best_received_g = received[i]
		# Set g to LOWEST value
		if best_received_g[-1] < self.g[-1]:
			self.g = best_received_g

	def random_confidence(self):
		'''
		Randomly assigns confidence parameters c2 and c3 in the interval [0, cmax)
		Parameters
		----------
		None
		Returns
		-------
		(c2, c3)
		c2 : np.ndarray
		The confidence in the Particle's p-value. The array has the same
		dimensions as the problem.
		c3 : np.ndarray
		The confidence in the particle's g-value. The array has the same
		dimensions as the problem.
		'''
		c2 = np.inf*np.ones(self.dim)
		c3 = np.inf*np.ones(self.dim)

		for d in range(self.dim):
			c2[d] = np.random.uniform(0, self.cmax)
			c3[d] = np.random.uniform(0, self.cmax)
		return (c2, c3)

	# Compare the new position and the old p-value, and update p
	def update_p(self, value):
		'''
		Compares the new position and the old p-value, and updates p accordingly.
		If the new position is better than the old p-value, then the p-value is set
		to the new position and that position's value.
		Parameters
		----------
		value : number
		The value of the position that is compared to p
		Returns
		-------
		None
		'''

		if value < self.p[self.dim]:
			self.p[self.dim] = value
			self.p[0:self.dim] = self.pos

	def update_g(self, value):
		'''
		Updates the particle's g-value if the new position is better than the
		previously known g-value. Finishes by communicating with informants
		and updating the g-value again.
		Parameters
		----------
		value : number
		The value of the position that is compared to g
		Returns
		-------
		None
		'''
		if value < self.g[self.dim]:
			self.g[self.dim] = value
			self.g[0:self.dim] = self.pos
		self.communicate()

	def find_vel(self):
		'''
		Calculates the velocity of the Particle according to the 
		update equation from the PSO book.
		Since this is PSO(0), c1 is constant, and c2,c2 are chosen
		randomly from a rectangular probability distribution.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''
		c2, c3 = self.random_confidence()		
		possible_vel = self.c1*self.vel + \
			c2*(self.p[0:self.dim] - self.pos) + \
			c3*(self.g[0:self.dim] - self.pos)	

		# Constrain velocity
		smaller_than_vmax = possible_vel < self.vmax
		possible_vel = np.where(smaller_than_vmax, possible_vel, self.vmax)
		greater_than_neg_vmax = possible_vel > -self.vmax
		possible_vel = np.where(greater_than_neg_vmax, possible_vel, -self.vmax)
		self.vel = possible_vel
	
	def set_pos(self):
		'''
		Uses the calculated velocity to set the next position for a Particle
		while taking into account any constraints on the search space.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''
		if self.special_constraints == True:
			# Set is_initial_positions to False
			self.constraints_extra_arguments[0] = False
			next_pos, vel = eval(self.constraints_function)(self, self.constraints_extra_arguments)
			self.pos = next_pos
			self.vel = vel
		else:
			possible_pos = self.pos + self.vel
			in_search_area = self.is_in_search_area(possible_pos)
			self.pos = np.where(in_search_area, possible_pos, self.pos)
			self.vel = np.where(in_search_area, self.vel, 0)

	def is_in_search_area(self, possible_pos):
		'''
		Checks whether a position is inside the allowed search space given by xmin and xmax.
		Returns True or False.
		Parameters
		----------
		possible_pos : np.ndarray
		A possible position of the particle that may or may not be within the search area.
		Returns
		-------
		is_allowed : bool
		True if possible_pos is inside the search area. False if possible_pos is not inside
		the search area.
		'''
		smaller_than_xmax = possible_pos <= self.xmax
		greater_than_xmin = possible_pos >= self.xmin
		is_allowed = np.zeros((len(self.xmax), 2))
		is_allowed[:,0] = smaller_than_xmax
		is_allowed[:,1] = greater_than_xmin
		is_allowed = np.all(is_allowed, axis=1)

		return is_allowed

	def step(self):
		'''
		Moves a Particle from one position to another while taking into account
		any constraints.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''
		value = eval(self.fn_name)(self.pos)
		self.update_p(value)
		self.update_g(value)
		self.find_vel()
		self.set_pos()

###################################################################

# Artificial Neural Network

def model(x, unpacked_params):
	l1,b1,l2,b2 = unpacked_params
	y = torch.nn.functional.linear(x,l1,b1)
	y = torch.relu(y)
	y = torch.nn.functional.linear(y,l2,b2)
	y = torch.log_softmax(y,dim=0)
	return y

def unpack_params(params, layers=[(50,4), (9,50)]):
	params = torch.from_numpy(params).float()
	unpacked_params = []
	e = 0
	for i,l in enumerate(layers):
		s,e = e,e+np.prod(l)
		weights = params[s:e].view(l)
		s,e = e,e+l[0]
		bias = params[s:e]
		unpacked_params.extend([weights,bias])
	return unpacked_params

def evaluation_function(vector):
	env = RobotArmGame()
	done = False
	state = torch.from_numpy(env.reset()).float()
	score = 0
	t = 0
	while not done:
		if t < 6000:
			params = unpack_params(vector)
			probs = model(state, params)
			action = torch.distributions.Categorical(probs=probs).sample()
			state_, reward, done, info = env.step(action.item())
			state = torch.from_numpy(state_).float()
			t += 1
		else:
			done = True
	score = t
	return score

###################################################################

# Training the ANN for the robot arm problem

fn_info = {
	"fn_name":'evaluation_function',
	"optimal_f":0, # We want to reduce the time to 0
	"dim":709,
	"xmin":np.ones(709)*-10, "xmax":np.ones(709)*10,
	"param_is_integer":np.zeros(709),
	"special_constraints":False, # N,t,r are related through the number of evaluations
	"constraints_function":None,
	"constraints_extra_arguments":[],
	"show_animation":False,
	"disable_progress_bar":False,
	"disable_printing":True,
	"get_parameters_from":"average p-values"
	}

constants = {'phi': 2.4, 'N': 9, 'k': 3, 'time_steps': 100, 'repetitions': 3}

def train(N=63, time_steps=100, repetitions=1):
	'''
	Trains the neural network using particle swarm optimisation.
	Returns a vector of the best weights and biases and the time with that configuration
	Inputs
	------
	N : int
	Number of particles
	time_steps : int
	repetitions : int
	Returns
	-------
	(best_position, best_f)
	best_position : np.ndarray
	A vector of 709 elements for the neural network
	best_f : the resulting time with that configuration
	'''
	constants['N'] = N
	constants['time_steps'] = time_steps
	constants['repetitions'] = repetitions
	experiment = Experiment(constants=constants, fn_info=fn_info)
	best_position, best_f, _ = experiment.run()
	best_parameters = best_position
	value = best_f
	return best_parameters, value