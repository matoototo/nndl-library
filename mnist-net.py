from network import Network, Layer
from trainer import Trainer
from functions import sigmoid, MSE, CrossEntropy, softmax, L2Reg
from numpy import array, reshape
from mnist import MNIST
from matplotlib import pyplot
from scipy.ndimage import rotate

# An example using the NNDL library on the MNIST dataset.
# It trains two networks, both initialized with the same weights.
# The only difference is in the activation function used in the final layer.

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

colours = ["b", "g", "r", "c", "m"]

l2 = L2Reg(5.0)
net1 = Network([Layer(784, 100, sigmoid, l2), Layer(100, 10, softmax, l2)])
net1.save('./weights/tmp.pkl')

net2 = Network.load('./weights/tmp.pkl')
net2.layers[-1].act = sigmoid

nets = [net1, net2]
for i in range(2):
	trainer = Trainer(nets[i], CrossEntropy, 0.5, 10, images, labels, images_test, labels_test)
	data = trainer.SGD(100, tr_acc = True, tr_loss = True)
	y_points = [point["epoch"] for point in data["test"]]
	x_points = [point["acc"] for point in data["test"]]
	pyplot.plot(y_points, x_points, colours[i])

pyplot.show()
