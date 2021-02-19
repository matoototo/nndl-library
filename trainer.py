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
		self.network.generate_masks(True) # set dropout masks
		a, z = self.network.infer(x)
		delta = self.loss.delta(a[-1], y, z[-1])
		nablas_w = [outer(a[-2], delta).T]
		deltas = [delta]
		n = len(self.network.layers)
		layers = self.network.layers # makes delta term easier to read
		for l, w_in, act in zip(range(n-1, 0, -1), z[::-1][1:], a[::-1][2:]):
			delta = (layers[l].weights.T @ deltas[-1]) * sigmoid_prime(w_in) * layers[l-1].dropout_mask
			deltas.append(delta)
			nablas_w.append(outer(act, delta).T)
		return nablas_w[::-1], deltas[::-1], self.stats(a[-1], y)

	def train_batch(self, batch_x, batch_y):
		nablas_w = [zeros_like(l.weights) for l in self.network.layers]
		nablas_b = [zeros_like(l.biases) for l in self.network.layers]
		correct = loss = 0
		for x, y in zip(batch_x, batch_y):
			d_w, d_b, (c, l) = self.backprop(x, y)
			correct += c
			loss += l
			nablas_w = [a+b for a, b in zip(d_w, nablas_w)]
			nablas_b = [a+b for a, b in zip(d_b, nablas_b)]
		self.gradient_descent(self.div_arrs(nablas_w, len(batch_x)), self.div_arrs(nablas_b, len(batch_x)))
		return (correct, loss)

	def SGD(self, epochs, report):
		batches_x = [array(self.x[i : i + self.batchsize]) for i in range(0, len(self.x), self.batchsize)]
		batches_y = [array(self.y[i : i + self.batchsize]) for i in range(0, len(self.y), self.batchsize)]
		batches_x, batches_y = self.shuffle_together(batches_x, batches_y)
		info = {"train": [], "test": []}
		for epoch in range(epochs):
			i = 0
			loss = correct = 0
			for batch_x, batch_y in zip(batches_x, batches_y):
				c, l = self.train_batch(batch_x, batch_y)
				correct += c
				loss += l
				report(i, self.batchsize, len(self.x))
				i += 1
			info["train"].append({"epoch": epoch, "acc": 100*correct/len(self.x), "loss": loss/len(self.x)})
			info["test"].append(self.evaluate(epoch))
			self.report_stats(info)
		return info

	def evaluate(self, epoch):
		loss = correct = 0
		self.network.generate_masks(False) # remove dropout
		for x, y in zip(self.x_test, self.y_test):
			a, _ = self.network.infer(x)
			loss += self.loss.loss(a[-1], y)
			correct += argmax(a[-1])==argmax(y)
		acc = 100*correct / len(self.x_test)
		loss = loss / len(self.x_test)
		return {"epoch": epoch+1, "acc": acc, "loss": loss}

	def stats(self, x, y):
		return (argmax(x)==argmax(y), self.loss.loss(x, y))

	@staticmethod
	def report_stats(info):
		test = info["test"][-1]
		train = info["train"][-1]
		print("\x1b[2K") # clear line
		print("Epoch", test["epoch"]+1)
		print("Test Accuracy:", test["acc"], "%")
		print("Test Loss:", round(test["loss"], 4))
		print("Train Accuracy:", train["acc"], "%")
		print("Train Loss:", round(train["loss"], 4))

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