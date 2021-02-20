from network import Network, Layer
from trainer import Trainer
from functions import sigmoid, MSE, CrossEntropy, softmax, L2Reg
from numpy import array
from mnist import MNIST
from matplotlib import pyplot

# An example using the nndl library on the MNIST dataset.

def one_hot(labels):
	ys = []
	for label in labels:
		y = [0] * 10
		y[label] = 1.0
		ys.append(array(y))
	return array(ys)

mnist = MNIST("./mnist")
mnist.gz = True
images, labels = mnist.load_training()
images_test, labels_test = mnist.load_testing()

images = array(images) / 255.
images_test = array(images_test) / 255.
labels = one_hot(labels)
labels_test = one_hot(labels_test)

colours = ["b", "g", "r", "c", "m"]

l2 = L2Reg(0.5)
net1 = Network([Layer(784, 50, sigmoid, l2, 0.0), Layer(50, 10, softmax, l2)])
nets = [net1]
for i in range(1):
	trainer = Trainer(nets[i], CrossEntropy, 0.5, 10, images, labels, images_test[0:1000], labels_test[0:1000])
	data = trainer.SGD(100, tr_acc = False, tr_loss = False)
	y_points = [point["epoch"] for point in data["test"]]
	x_points = [point["acc"] for point in data["test"]]
	pyplot.plot(y_points, x_points, colours[i])

pyplot.show()
