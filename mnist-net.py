from network import Network, Layer
from trainer import Trainer
from functions import sigmoid, MSE, CrossEntropy, softmax, L2Reg
from numpy import array, reshape
from mnist import MNIST
from matplotlib import pyplot
from scipy.ndimage import rotate

# An example using the nndl library on the MNIST dataset.

def one_hot(labels):
	ys = []
	for label in labels:
		y = [0] * 10
		y[label] = 1.0
		ys.append(array(y))
	return array(ys)

def augment(images, labels, start = 10, end = 20, inc = 10):
	aug_images = []
	aug_labels = []
	for image, label in zip(images, labels):
		aug_images.append(image)
		aug_labels.append(label)
		for i in range(start, end, inc):
			rotated = rotate(reshape(image, (28, 28)), i, reshape=False)
			aug_images.append(reshape(rotated, (784)))
			aug_labels.append(label)
	return (array(aug_images), array(aug_labels))

mnist = MNIST("./mnist")
mnist.gz = True
images, labels = mnist.load_training()
images_test, labels_test = mnist.load_testing()

images = array(images) / 255.
images_test = array(images_test) / 255.
labels = one_hot(labels)
labels_test = one_hot(labels_test)

aug_images, aug_labels = augment(images[0:1000], labels[0:1000], start=-5, end=15, inc=10)

colours = ["b", "g", "r", "c", "m"]

l2 = L2Reg(0.5)
net1 = Network([Layer(784, 30, sigmoid, l2, 0.0), Layer(30, 10, softmax, l2)])
l2 = L2Reg(1.5)
net2 = Network([Layer(784, 30, sigmoid, l2, 0.0), Layer(30, 10, softmax, l2)])
nets = [net1, net2]
for i in range(2):
	if i == 0: trainer = Trainer(nets[i], CrossEntropy, 0.5, 10, images[0:1000], labels[0:1000], images_test[0:2500], labels_test[0:2500])
	if i == 1: trainer = Trainer(nets[i], CrossEntropy, 0.5, 10, aug_images[0:3000], aug_labels[0:3000], images_test[0:2500], labels_test[0:2500])
	data = trainer.SGD(300, tr_acc = False, tr_loss = False)
	y_points = [point["epoch"] for point in data["test"]]
	x_points = [point["acc"] for point in data["test"]]
	pyplot.plot(y_points, x_points, colours[i])
	x_points = [point["acc"] for point in data["train"]]
	pyplot.plot(y_points, x_points, colours[i]+"--")

pyplot.show()
