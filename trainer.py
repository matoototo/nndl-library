from network import Network
from functions import sigmoid_prime, Loss
from numpy import tile, outer, array, argmax, zeros_like
from random import shuffle


class Trainer:
	"""A trainer class that controls the training of the network."""
	def __init__(self, network: Network, loss: Loss, lr: float, batchsize: int, x, y, x_test, y_test):
		"""loss (Loss) : the loss function to use
		lr (float) : learning rate"""
		self.network = network
		self.loss = loss
		self.lr = lr
		self.batchsize = batchsize
		self.x = x
		self.y = y
		self.x_test = x_test
		self.y_test = y_test

	def gradient_descent(self, nablas_w, nablas_b):
		for ndw, ndb, layer in zip(nablas_w, nablas_b, self.network.layers):
			layer.weights -= self.lr * (ndw + layer.reg.partial_w(layer)/len(self.x))
			layer.biases -= self.lr * ndb

	def backprop(self, x, y):
		a, z = self.network.infer(x)
		delta = self.loss.delta(a[-1], y, z[-1])
		nablas_w = [outer(a[-2], delta).T]
		deltas = [delta]
		for layer, w_in, act in zip(self.network.layers[::-1], z[::-1][1:], a[::-1][2:]):
			delta = (layer.weights.T @ deltas[-1]) * sigmoid_prime(w_in)
			deltas.append(delta)
			nablas_w.append(outer(act, delta).T)
		return nablas_w[::-1], deltas[::-1]

	def train_batch(self, batch_x, batch_y):
		nablas_w = [zeros_like(l.weights) for l in self.network.layers]
		nablas_b = [zeros_like(l.biases) for l in self.network.layers]
		for x, y in zip(batch_x, batch_y):
			d_w, d_b = self.backprop(x, y)
			nablas_w = [a+b for a, b in zip(d_w, nablas_w)]
			nablas_b = [a+b for a, b in zip(d_b, nablas_b)]
		self.gradient_descent(self.div_arrs(nablas_w, len(batch_x)), self.div_arrs(nablas_b, len(batch_x)))

	def SGD(self, epochs, report):
		batches_x = [array(self.x[i : i + self.batchsize]) for i in range(0, len(self.x), self.batchsize)]
		batches_y = [array(self.y[i : i + self.batchsize]) for i in range(0, len(self.y), self.batchsize)]
		batches_x, batches_y = self.shuffle_together(batches_x, batches_y)
		info = []
		for epoch in range(epochs):
			i = 0
			for batch_x, batch_y in zip(batches_x, batches_y):
				self.train_batch(batch_x, batch_y)
				report(i, self.batchsize, len(self.x))
				i += 1
			info.append(self.evaluate(epoch))
		return info

	def evaluate(self, epoch):
		loss = correct = 0
		for x, y in zip(self.x_test, self.y_test):
			a, _ = self.network.infer(x)
			loss += self.loss.loss(a[-1], y)
			correct += argmax(a[-1])==argmax(y)
		acc = 100*correct / len(self.x_test)
		loss = loss / len(self.x_test)
		print("\x1b[2K") # clear line
		print(f"Epoch {epoch+1}")
		print("Accuracy:", acc, "%")
		print("Loss:", round(loss, 4))
		return {"epoch": epoch+1, "acc": acc, "loss": loss}

	@staticmethod
	def report_train(i, batchsize, length):
		if (not i%20): print(str(i*batchsize) + "/" + str(length), end='\r')

	@staticmethod
	def shuffle_together(x, y):
		xy = list(zip(x, y))
		shuffle(xy)
		x, y = zip(*xy)
		return x, y

	@staticmethod
	def div_arrs(arrays, divisor):
		return [x/divisor for x in arrays]