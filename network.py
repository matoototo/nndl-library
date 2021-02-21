from numpy import random, dot, array, stack, tile
from random import random as rfloat, seed
from functions import softmax
import pickle


class Layer:
	def __init__(self, inputs, size, activation, reg, dropout = 0.0):
		"""inputs (int) : number of neurons in the previous layer
		size (int) : number of neurons in the current layer
		activation (func) : function used as the activation
		reg (Reg) : used regularization
		dropout (float) : dropout factor, [0, 1]"""
		self.weights = random.normal(0, 1, (size, inputs))
		self.biases = random.normal(0, 1, (size))
		self.act = activation
		self.reg = reg
		self.dropout = dropout
		self.dropout_mask = self.generate_mask()

	def input_size(self):
		return len(self.weights[0])

	def generate_mask(self, train = True):
		n = len(self.biases)
		if not train: # when not training, don't do anything
			return tile(1, n)
		mask = tile(0, n)
		for i in range(n):
			if rfloat() > self.dropout:
				mask[i] = 1/(1-self.dropout)
		return mask

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
			a.append(layer.act(z[-1]) * layer.dropout_mask)
		return a, z

	def generate_masks(self, train = True):
		for layer in self.layers:
			layer.dropout_mask = layer.generate_mask(train)

	def save(self, path):
		with open(path, "wb") as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(path):
		with open(path, "rb") as input:
			return pickle.load(input)
