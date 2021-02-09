from numpy import exp, sum, log
from numpy.linalg import norm


def sigmoid(z):
	return 1 / (1 + exp(-z))


def softmax(z):
	return exp(z) / sum(exp(z), axis=0)


def sigmoid_prime(z):
	sig_z = sigmoid(z)
	return sig_z * (1 - sig_z)


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

	@staticmethod
	def delta_term(w):
		return 1


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

	@staticmethod
	def delta_term(w):
		return sigmoid_prime(w)


class CrossEntropy(Loss):
	@staticmethod
	def loss(x, y):
		assert len(x) == len(y)
		return -sum(y * log(x) + (1 - y) * log(1 - x)) / len(x)

	@staticmethod
	def partial_a(a, y):
		return a - y

	@staticmethod
	def delta(a, y, z):
		return CrossEntropy.partial_a(a, y)
