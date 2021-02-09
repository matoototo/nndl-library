from network import Network
from trainer import Trainer
from functions import sigmoid, MSE, CrossEntropy
from numpy import array
from mnist import MNIST
from numpy import array

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

labels = one_hot(labels)
labels_test = one_hot(labels_test)

net = Network(array([784, 30, 10]), sigmoid)
trainer = Trainer(net, CrossEntropy, 2, 64, images, labels, images_test, labels_test)

trainer.SGD(10, Trainer.report_train)
