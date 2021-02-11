from network import Network, Layer
from trainer import Trainer
from functions import sigmoid, MSE, CrossEntropy, softmax
from numpy import array
from mnist import MNIST

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

images = array(images)/255.
images_test = array(images_test)/255.
labels = one_hot(labels)
labels_test = one_hot(labels_test)

net = Network([Layer(784, 10, sigmoid), Layer(30, 10, softmax)])
trainer = Trainer(net, CrossEntropy, 4.0, 64, images, labels, images_test, labels_test)

trainer.SGD(10, Trainer.report_train)
