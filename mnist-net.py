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


# run two runs: one with reg and one without and plot the results

for i in range(2):
	l2 = L2Reg(i * 0.5)
	net = Network([Layer(784, 30, sigmoid, l2), Layer(30, 10, sigmoid, l2)])
	trainer = Trainer(net, CrossEntropy, 0.5, 10, images[0:1000], labels[0:1000], images_test, labels_test)

	data = trainer.SGD(400, Trainer.report_train)
	x_points = [point["acc"] for point in data]
	y_points = [point["epoch"] for point in data]
	pyplot.plot(y_points, x_points, colours[i])

pyplot.show()
