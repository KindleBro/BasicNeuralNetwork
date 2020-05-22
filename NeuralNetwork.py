"""
Coded Neural Network based on the code found below:
https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
22/05/2020 @ 13:35
Kieran Findlay
"""

# imports
from ActivationFunction import *


# class
class NeuralNetwork:
	"""
	Class based Neural network.
	"""
	
	def __init__(self, name: str, should_debug_log: bool = False):
		"""
		Blank initialization to allow loading of weights & bias' using setup function.
		:param architecture: A list of dictionaries of neurons. Should be formatted like this:
							[ {"input_dim": 1, "output_dim": 2, "activation": "relu"},
							{"input_dim": 2, "output_dim": 1, "activation": "sigmoid"} ]
	    :param number_of_layers: Derived from the length of architecture.
		:param parameter_values: A dictionary that will contain each neurons weightings and bias'.
		:param learning_rate: How fast the neural network should learn. Too fast will never converge,
		                      too slow will take too long.
        :param adaptive_lr_acc_bound: The bound to switch off adaptive learning rate.
		:param bounded_lr: When the adaptive learning rate is switched off, this value is set.
        :param epochs: How many times to run through the training data.
		"""
		
		self.name = name
		self.architecture = None
		self.number_of_layers = None
		self.parameter_values = None
		
		# TODO: Make a better adaptive LR start and stop.
		self.learning_rate = None
		self.adaptive_lr_acc_bound = 0.9
		self.bounded_lr = 1
		self.epochs = None
		
		self.console_log = list()
		self.should_debug_log = should_debug_log
		
		self.debug_log("Init {} neural network.".format(self.name))
	
	# Extended init functions.
	def new(self, architecture: list, learning_rate: float = 1, epochs: int = 1):
		"""
		Setup a new neural network.
		Check __init__ for details on each value.
		"""
		self.setup(architecture, learning_rate, epochs)
		self.init_parameters()
		self.log("Completed '{}' neural network initialization.".format(self.name))
	
	def load(self, architecture: list, parameter_values: dict, learning_rate: int):
		"""
		Load an existing model.
		Check __init__ for details on each value.
		"""
		# Error checking.
		if len(architecture) != (len(parameter_values) / 2):  # Parameter list must have 1 weight and 1 bias per neuron.
			self.log("Architecture shape does not match parameter shape")
			self.log("Please ensure you have the correct size architecture & parameter shape: {} vs {} / 2"
			         .format(len(architecture), len(parameter_values)))
		
		# Set class values.
		self.setup(architecture, learning_rate)
		self.parameter_values = parameter_values
		
		self.log("Successfully loaded architecture of size {} with parameter list of size {}"
		         .format(len(architecture), len(parameter_values)))
	
	def setup(self, architecture: list, learning_rate: float = 1, epochs: int = 1):
		"""
		Post initialization of parts of the neural network.
		Check __init__ for details on each value.
		"""
		self.architecture = architecture
		self.number_of_layers = len(architecture)
		self.learning_rate = learning_rate
		self.epochs = epochs
		
		self.debug_log("architecture: {}".format(self.architecture), indent=2)
		self.debug_log("number_of_layers: {}".format(self.number_of_layers), indent=2)
		self.debug_log("learning_rate: {}".format(self.learning_rate), indent=2)
	
	def init_parameters(self):
		"""
		Init the neural networks parameter list.
		"""
		self.parameter_values = dict()
		
		# For each layer...
		for index, layer in enumerate(self.architecture):
			# Get the layer values.
			layer_index = index + 1
			layer_input_size = layer["input_dim"]
			layer_output_size = layer["output_dim"]
			
			# Generate random values for weights and bias'.
			self.parameter_values['W' + str(layer_index)] = np.random.randn(
				layer_output_size,
				layer_input_size) * 0.1  # randn gets random values using standard normal distribution.
			self.parameter_values['b' + str(layer_index)] = np.random.randn(
				layer_output_size, 1) * 0.1
		
		self.log("Parameter list initialized.")
		self.debug_log("Parameter list: {}".format(self.parameter_values))
		return
	
	# Error checking
	@staticmethod
	def get_activation_function(activation):
		"""
		Get the activation function of a layer. Can either check if the current activation function is a valid input or
		get the activation function of a string input.
		:param activation: Class or string input to check or get.
		:return: A valid activation function.
		"""
		if isinstance(activation, ActivationFunction):
			activation_func = activation
		elif isinstance(activation, str):  # ... otherwise, get and set.
			if activation is "relu":
				activation_func = ReLu()
			elif activation is "sigmoid":
				activation_func = Sigmoid()
			elif activation is "swish":
				activation_func = Swish()
			else:
				raise Exception('Non-supported activation function')
		else:
			raise Exception('Non-supported activation function')
		
		return activation_func
	
	# Forward propagation
	def single_layer_forward_propagation(self, A_prev, W_curr, b_curr,
	                                     activation: ActivationFunction = ReLu()):
		"""
		Performs a single layer of forward propagation
		:param A_prev: Previous layers output.
		:param W_curr: This layers weightings.
		:param b_curr: This layers bias'.
		:param activation: This layers activation function. Can either be a class of Activation function, or a string
		repr of a desired activation function.
		:return: This layers output along with this layers pre-output (Useful for derivatives and other math later).
		"""
		Z_curr = np.dot(W_curr, A_prev) + b_curr
		
		# Check and get the given activation function.
		activation_func = self.get_activation_function(activation)
		
		return activation_func.forward(Z_curr), Z_curr
	
	def full_forward_propagation(self, input_layer: np.ndarray):
		"""
		Performs forward propagation across the entire Neural network.
		:param input_layer: The input for the neural network. Should be a numpy array.
		:return: Neural networks' prediction alongside the "memory" of each layers output and pre-output.
		"""
		memory = {}
		A_curr = input_layer
		
		# For each layer...
		for index, layer in enumerate(self.architecture):
			# Setup for the loop.
			layer_index = index + 1
			A_prev = A_curr
			
			# Get the get activation function, weight, and bias' for this layer
			activation_function_cur = layer["activation"]
			W_curr = self.parameter_values["W" + str(layer_index)]
			b_curr = self.parameter_values["b" + str(layer_index)]
			
			# Perform one layer of forward propagation.
			A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activation_function_cur)
			
			# Store values of result.
			memory["A" + str(index)] = A_prev
			memory["Z" + str(layer_index)] = Z_curr
		
		return A_curr, memory
	
	# Backward propagation
	def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev,
	                                      activation: ActivationFunction = ReLu()):
		"""
		Performs a single layer of backwards propagation.
		:param dA_curr: Current layers output.
		:param W_curr: Current layers weightings.
		:param b_curr: Current layers bias'. Unused but left in case of further development.
		:param Z_curr: Current layers pre-output.
		:param A_prev: The previous layers output.
		:param activation: This layers activation function. Can either be a class of Activation function, or a string
		repr of a desired activation function.
		:return: Derived values for this layers output, weightings, and bias'.
		"""
		m = A_prev.shape[1]  # Get size of layer.
		
		# Check and get the given activation function.
		activation_func = self.get_activation_function(activation)
		
		# Derive each value for full backwards propagation.
		dZ_curr = activation_func.backward(dA_curr, Z_curr)  # dZ = dA * pre-output(Z)
		dW_curr = (1 / m) * np.dot(dZ_curr, A_prev.T)  # dW = (1/m) * dZ * transpose(dA[--i])
		db_curr = (1 / m) * np.sum(dZ_curr, axis=1, keepdims=True)  # dB = (1/m) * sum(dZ)
		dA_prev = np.dot(W_curr.T, dZ_curr)  # dA[--i] = transpose(W) * dZ
		
		return dA_prev, dW_curr, db_curr
	
	def full_backward_propagation(self, prediction, output, memory: dict):
		"""
		Performs backward propagation across the entire Neural network.
		:param prediction: The predicted output of the neural network for the given input.
		:param output: Expected output for the given input.
		:param memory: Each layers pre-output and output for the given input.
		:return: Delta values to update each layers weightings and bias'.
		"""
		# Prep for loop.
		delta_values = dict()
		m = output.shape[1]
		Y_hat = prediction
		Y = output.reshape(Y_hat.shape)
		
		# Derivation of cost function: -( (output / prediction) - (-output) / (-prediction) )
		dA_prev = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
		
		# For each layer, starting at the output layer...
		for layer_index_prev, layer in reversed(list(enumerate(self.architecture))):
			# Get layers index & activation function
			layer_index_curr = layer_index_prev + 1
			activation_function_current = self.get_activation_function(layer["activation"])
			
			dA_curr = dA_prev
			
			# Get each layers values for backwards propagation.
			A_prev = memory["A" + str(layer_index_prev)]
			Z_curr = memory["Z" + str(layer_index_curr)]
			W_curr = self.parameter_values["W" + str(layer_index_curr)]
			b_curr = self.parameter_values["b" + str(layer_index_curr)]
			
			# Perform a single layers backwards propagation.
			dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
				dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_function_current)
			
			# Store each layers deltas
			delta_values["dW" + str(layer_index_curr)] = dW_curr
			delta_values["db" + str(layer_index_curr)] = db_curr
		
		return delta_values
	
	# Updates
	def update_parameters(self, delta_values: dict):
		"""
		For each layer, update its weightings and bias'.
		:param delta_values: How much each layers prediction was off by.
		:return:
		"""
		
		for index, layer in enumerate(self.architecture):
			layer_index = index + 1
			self.parameter_values["W" + str(layer_index)] -= self.learning_rate * delta_values["dW" + str(layer_index)]
			self.parameter_values["b" + str(layer_index)] -= self.learning_rate * delta_values["db" + str(layer_index)]
		
		return
	
	# Training
	def adaptive_learning_rate(self, this_cost: np.ndarray, last_cost: np.ndarray):
		"""
		Automatically update the learning rate to increase learning speed.
		:param this_cost: The cost of this epoch.
		:param last_cost: The cost of last epoch
		:return: The difference between this cost and last cost (In percent).
		"""
		if this_cost < last_cost:
			multiplier = 1.2
		else:
			multiplier = 0.8
		self.learning_rate *= multiplier
		
		return multiplier
	
	def train(self, input_array: np.ndarray, output_array: np.ndarray, batch_size: int):
		"""
		Train the neural network with the given inputs and outputs.
		:param input_array: The inputs to train the neural network on.
		:param output_array: The outputs to compare the predictions to.
		:param batch_size: THe size of each batch.
		:return: The parameter dictionary alongside the cost history and the accuracy history
		"""
		cost_history = list()
		accuracy_history = list()
		learning_rate_history = list()
		
		do_adapt_lr = False
		
		# For each epoch...
		for i in range(self.epochs):
			last_batch_index = 0
			
			# For each batch...
			# TODO: Non-uniform batch size will miss final batch. Fix.
			# TODO: Move neural network training to separate function to avoid excessive function size.
			for j in range(batch_size, len(input_array[0])+1, batch_size):
				batch_array_in = input_array[0][last_batch_index:j]
				batch_array_out = output_array[0][last_batch_index:j]
				
				# Init batch array and get values for it.
				batch_input = np.zeros((1, batch_size))
				batch_output = np.zeros((1, batch_size))
				for k in range(len(batch_array_in)):
					batch_input[0][k] = batch_array_in[k]
					batch_output[0][k] = batch_array_out[k]
				last_batch_index = j
				
				# Get the neural networks prediction.
				prediction, cache = self.full_forward_propagation(batch_input)
				
				# Get neural networks errors.
				delta_values = self.full_backward_propagation(prediction, batch_output, cache)
				self.update_parameters(delta_values)  # Update parameters.
				
				# Calculate the cost and accuracy of this epoch.
				cost = self.get_cost_value(prediction, batch_output)
				accuracy = self.get_accuracy_value(prediction, batch_output)
				
				# Try adapt learning rate
				if accuracy < self.adaptive_lr_acc_bound:
					if do_adapt_lr:
						percent_increase = self.adaptive_learning_rate(cost, cost_history[-1])
						learning_rate_history.append(percent_increase)
					else:
						do_adapt_lr = True
				else:
					self.learning_rate = self.bounded_lr
				
				# Store cost and accuracy.
				cost_history.append(cost)
				accuracy_history.append(accuracy)
				
				# Log the cost and accuracy of this batch.
				self.full_feed_log((j // batch_size)-1, cost, accuracy)
			
			# Log epoch.
			self.full_feed_log(i, cost_history[-1], accuracy_history[-1], is_batch=False)
		
		print("========")
		return self.parameter_values, cost_history, accuracy_history
	
	# Getters
	@staticmethod
	def get_cost_value(Y_hat, Y) -> np.ndarray:
		"""
		Calculate the cost of the neural network.
		:param Y_hat: The neural networks prediction.
		:param Y: The expected output
		:return:
		"""
		# Get the shape of the layer.
		m = Y_hat.shape[1]
		
		# Cost = -(1/m) * ( output * log(prediction) + (1-output) * log(1-prediction) )
		cost = -(1/m) * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
		
		return np.squeeze(cost)
	
	def get_accuracy_value(self, Y_hat, Y) -> np.ndarray:
		"""
		Calculate the accuracy of the neural network.
		:param Y_hat: The neural networks prediction.
		:param Y: The expected output
		:return:
		"""
		Y_hat_ = self.prob_to_class(Y_hat)
		
		return (Y_hat_ == Y).all(axis=0).mean()
	
	@staticmethod
	def prob_to_class(probability):
		probability_ = np.copy(probability)
		probability_[probability_ > 0.5] = 1
		probability_[probability_ <= 0.5] = 0
		return probability_
	
	# Export
	def export(self):
		"""
		Exports relevant information about the neural network.
		:return: Tuple of architecture, parameter values, learning rate and epochs.
		"""
		data_list = [self.name, self.architecture, self.parameter_values]
		data = tuple(data_list)
		return data
	
	# Debug and print functions.
	def feed_log_values(self, cost, accuracy, indents: int = 1):
		# Log the cost and accuracy of this epoch.
		self.log("Cost: {}".format(cost), indents)
		self.log("Accuracy: {}".format(accuracy), indents)
		self.log("Learning rate: {}".format(self.learning_rate), indents)
	
	def full_feed_log(self, index: int, cost, accuracy, is_batch: bool = True):
		indents = 1
		if is_batch:
			if self.should_debug_log:
				indents += 1
				printable_index = index + 1
				print("{}----".format(" "*indents))
				self.log("Batch {} info:".format(printable_index), indents)
				indents += 1
				self.feed_log_values(cost, accuracy, indents=indents)
		else:
			printable_index = index + 1
			print("========")
			self.log("Epoch {} info:".format(printable_index), indents)
			indents += 1
			self.feed_log_values(cost, accuracy, indents=indents)
	
	def debug_log(self, message, indent: int = 1):
		"""
		Debug logging for testing.
		:param message: The object to debug.
		:param indent: How much to indent the debug message by.
		:return:
		"""
		if self.should_debug_log:
			self.log("DEBUG >> {}".format(message, indent))
	
	def log(self, message, indent: int = 1):
		"""
		Log a message.
		:param message: Message to log.
		:param indent: How many times to indent the message.
		:return:
		"""
		self.console_log.append(message)
		self.print(message, indent)
	
	@staticmethod
	def print(message, indent: int):
		"""
		Formatted print statement to make things pretty.
		:param message: Object to print.
		:param indent: How many times to indent the message.
		:return:
		"""
		print("{}> {}".format("  " * indent, message))
