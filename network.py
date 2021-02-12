from numpy import random, dot, array, stack
from functions import softmax


class Layer:
	def __init__(self, inputs, size, activation, reg):
		"""inputs (int) : number of neurons in the previous layer
		size (int) : number of neurons in the current layer"""
		self.weights = random.normal(0, 1, (size, inputs))
		self.biases = random.normal(0, 1, (size))
		self.act = activation
		self.reg = reg

	def input_size(self):
		return len(self.weights[0])


class Network:
	def __init__(self, layers):
		"""layers (array(Layer)) : an array of Layers comprising the network"""
		self.layers = layers

	def infer(self, a):
		"""a (np.array<float>) : an array of floats -- the input to the network"""
		assert len(a) == self.layers[0].input_size()
		a, z = [a], []
		for layer in self.layers:
			z.append((layer.weights @ a[-1]) + layer.biases)
			a.append(layer.act(z[-1]))
		return a, array(z, dtype=object)
