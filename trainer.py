from network import Network
from functions import sigmoid_prime, Loss
from numpy import tile, outer, array, argmax, zeros_like, zeros
from random import shuffle


class Trainer:
	"""A trainer class that controls the training of the network."""
	def __init__(self, network: Network, loss: Loss, lr: float, batchsize: int, x, y, x_test, y_test, momentum: float = 0.0, nesterov = True):
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
		self.momentum = momentum
		self.vws = [zeros_like(layer.weights) for layer in self.network.layers]
		self.vbs = [zeros_like(layer.biases) for layer in self.network.layers]
		self.nesterov = nesterov

	def gradient_descent(self, nablas_w, nablas_b):
		term = 1+self.momentum if self.nesterov else 1
		self.vws = [vw*self.momentum - term*ndw*self.lr for vw, ndw in zip(self.vws, nablas_w)]
		self.vbs = [vb*self.momentum - term*ndb*self.lr for vb, ndb in zip(self.vbs, nablas_b)]
		for layer, vw, vb in zip(self.network.layers, self.vws, self.vbs):
			layer.weights -= self.lr*layer.reg.partial_w(layer)/len(self.x) - vw
			layer.biases -= -vb
		if self.nesterov:
			self.vws = [vw + self.lr*ndw for vw, ndw in zip(self.vws, nablas_w)]
			self.vbs = [vb + self.lr*ndb for vb, ndb in zip(self.vbs, nablas_b)]

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
		stats = array((0.0, 0.0))
		for x, y in zip(batch_x, batch_y):
			d_w, d_b, d_s = self.backprop(x, y)
			stats += d_s
			nablas_w = [a+b for a, b in zip(d_w, nablas_w)]
			nablas_b = [a+b for a, b in zip(d_b, nablas_b)]
		self.gradient_descent(self.div_arrs(nablas_w, len(batch_x)), self.div_arrs(nablas_b, len(batch_x)))
		return stats


	def SGD(self, epochs, early_stop = float("inf"), **kwargs):
		batches_x = [array(self.x[i : i + self.batchsize]) for i in range(0, len(self.x), self.batchsize)]
		batches_y = [array(self.y[i : i + self.batchsize]) for i in range(0, len(self.y), self.batchsize)]
		batches_x, batches_y = self.shuffle_together(batches_x, batches_y)
		info = {"train": [], "test": []}
		current_best = 0
		for epoch in range(epochs):
			i = 0
			stats = array((0.0, 0.0))
			for batch_x, batch_y in zip(batches_x, batches_y):
				# self.momentum = min(self.momentum_goal, 1-3/((epoch*len(self.x)+i*self.batchsize)/50000+5)) # momentum warmup
				d_s = self.train_batch(batch_x, batch_y)
				stats += d_s
				self.report_train(i, self.batchsize, len(self.x))
				i += 1
			info["train"].append({"epoch": epoch, "acc": 100*stats[0]/len(self.x), "loss": stats[1]/len(self.x)})
			info["test"].append(self.evaluate(epoch))
			self.report_stats(info, **kwargs)
			if (info["test"][epoch]["loss"] < info["test"][current_best]["loss"]): current_best = epoch
			elif (epoch-current_best == early_stop): break
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
		return {"epoch": epoch, "acc": acc, "loss": loss}

	def stats(self, x, y):
		return (argmax(x)==argmax(y), self.loss.loss(x, y))

	@staticmethod
	def report_stats(info, epoch = True, te_acc = True, te_loss = True, tr_acc = True, tr_loss = True):
		test = info["test"][-1]
		train = info["train"][-1]
		print("\x1b[2K") # clear line
		if (epoch): print("Epoch", test["epoch"]+1)
		if (te_acc): print("Test Accuracy:", round(test["acc"], 2), "%")
		if (te_loss): print("Test Loss:", round(test["loss"], 4))
		if (tr_acc): print("Train Accuracy:", round(train["acc"], 2), "%")
		if (tr_loss): print("Train Loss:", round(train["loss"], 4))

	@staticmethod
	def report_train(i, batchsize, length):
		if (not i%5): print(str(i*batchsize) + "/" + str(length), end='\r')

	@staticmethod
	def shuffle_together(x, y):
		xy = list(zip(x, y))
		shuffle(xy)
		x, y = zip(*xy)
		return x, y

	@staticmethod
	def div_arrs(arrays, divisor):
		return [x/divisor for x in arrays]
