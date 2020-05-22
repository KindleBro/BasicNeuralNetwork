"""
Various activation functions compiled into a class-based system for easy management
22/05/2020 @ 13:38
Kieran Findlay
"""


# imports
import numpy as np


# classes
class ActivationFunction:
	"""
	A base class for activation functions.
	"""
	
	def __init__(self):
		pass
	
	def forward(self, X):
		"""
		Override this function in child classes to return your forward value.
		:param X:
		:return:
		"""
		return
	
	def backward(self, dA, X):
		"""
		Override this function in child classes to return your backward value.
		:param dA:
		:param X:
		:return:
		"""
		return


class Sigmoid(ActivationFunction):
	
	def __init__(self):
		super().__init__()
	
	def forward(self, X):
		return 1 / (1 + np.exp(-X))
	
	def backward(self, dA, X):
		sig = self.forward(X)
		return dA * sig * (1 - sig)


class ReLu(ActivationFunction):
	
	def __init__(self):
		super().__init__()
	
	def forward(self, X):
		return np.maximum(0, X)
	
	def backward(self, dA, X):
		dX = np.array(dA, copy=True)
		dX[X <= 0] = 0
		return dX


class Swish(ActivationFunction):
	
	def __init__(self):
		super().__init__()
	
	def forward(self, X):
		return X / (1 + np.exp(-X))
	
	def backward(self, dA, X):
		sig = self.forward(X)
		return dA * X * sig * (1 - sig)
