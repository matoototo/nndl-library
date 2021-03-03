from numpy import exp, sum, log, dot, sign, max
from scipy.special import expit, softmax as sft
from numpy.linalg import norm


def sigmoid(z):
	return expit(z)  # same as below, just without overflow
	# return 1 / (1 + exp(-z))


def softmax(z):
	# return sft(z)  # same as below, just without overflow
	shifted = z-max(z)
	return exp(shifted) / sum(exp(shifted), axis=0)


def sigmoid_prime(z):
	sig_z = sigmoid(z)
	return sig_z * (1 - sig_z)


def softmax_prime(z):
	soft_z = softmax(z)
	return soft_z * (1 - soft_z)


class L2Reg:
	def __init__(self, lmbda):
		self.lmbda = lmbda

	def cost_term(self, net):
		return self.lmbda * sum([sum(layer.weights * layer.weights) for layer in net.layers])

	def partial_w(self, layer):
		return layer.weights * self.lmbda

class L1Reg:
	def __init__(self, lmbda):
		self.lmbda = lmbda

	def cost_term(self, net):
		return self.lmbda * sum([sum(layer.weights) for layer in net.layers])

	def partial_w(self, layer):
		return sign(layer.weights) * self.lmbda


class Loss:
	@staticmethod
	def loss(x, y):
		return

	@staticmethod
	def partial_a(a, y):
		return

	@staticmethod
	def delta(a, y, z):
		return


class MSE(Loss):
	@staticmethod
	def loss(x, y):
		assert len(x) == len(y)
		return 0.5 * norm(x - y) ** 2

	@staticmethod
	def partial_a(a, y):
		return a - y

	@staticmethod
	def delta(a, y, z):
		return MSE.partial_a(a, y) * sigmoid_prime(z)


class CrossEntropy(Loss):
	@staticmethod
	def loss(x, y):
		epsilon = 1e-5  # to avoid limited float precision causing log(0)
		assert len(x) == len(y)
		return -sum(y * log(x + epsilon) + (1 - y) * log(1 - x + epsilon)) / len(x)

	@staticmethod
	def partial_a(a, y):
		return a - y

	@staticmethod
	def delta(a, y, z):
		return CrossEntropy.partial_a(a, y)


class LogLikelihood(Loss):
	@staticmethod
	def loss(x, y):
		return dot(y.T, -log(x))

	@staticmethod
	def partial_a(a, y):
		return dot(y.T, -a)

	@staticmethod
	def delta(a, y, z):
		return a - y
