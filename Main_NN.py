"""
Main file for this neural network.
22/05/2020 @ 15:28
Kieran Findlay
"""

# imports
import numpy as np
from random import random
from NeuralNetwork import NeuralNetwork


# functions
def generate_nn_pair() -> tuple:
	"""
	NN should output 1 if above zero, or 0 if below zero.
	:return: Tuple of input to expected output.
	"""
	gen = 2 * random() - 1
	
	if gen > 0:
		out = 1
	else:
		out = 0
	
	return (gen, out)


# setup
iterations = 200
epochs = 3
batches_per_epoch = 4
learning_rate = 3
test_size = 5

# Generating inputs and outputs
inputs = np.zeros((1, iterations))
outputs = np.zeros((1, iterations))
for i in range(iterations):
	data = generate_nn_pair()
	inputs[0][i] = data[0]
	outputs[0][i] = data[1]

# Generating test set
test_in = np.zeros((1, test_size))
test_out = np.zeros((1, test_size))
for i in range(test_size):
	data = generate_nn_pair()
	test_in[0][i] = data[0]
	test_out[0][i] = data[1]

# main
if __name__ == "__main__":
	nn_architecture = [{"input_dim": 1, "output_dim": 2, "activation": "sigmoid"},
					   {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}]
	
	nn = NeuralNetwork("Test_Network", should_debug_log=False)
	nn.new(nn_architecture, learning_rate=learning_rate)
	parameter_values, cost_history, accuracy_history = nn.train(inputs, outputs,
	                                                            epochs=epochs, batch_size=batches_per_epoch)
	for i in range(test_size):
		prediction, _ = nn.full_forward_propagation(test_in[0][i])
		print("I: {:.2f}, ({} vs {:.2f})".format(test_in[0][i], test_out[0][i], prediction[0][0]))
	
































