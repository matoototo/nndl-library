from numpy import random, dot, array, stack
from functions import softmax


class Layer:
    def __init__(self, inputs, size, activation):
        """inputs (int) : number of neurons in the previous layer
        size (int) : number of neurons in the current layer"""
        self.weights = random.normal(0, 1, (size, inputs))
        self.biases = random.normal(0, 1, (size))
        self.act = activation

    def input_size(self):
        return len(self.weights[0])


class Network:
    def __init__(self, sizes, activation):
        """sizes (np.array<int>) : an array of integers denoting the sizes of each layer
        activation (method) : the activation function"""
        self.layers = []
        for i in range(1, len(sizes)):
            self.layers.append(Layer(sizes[i - 1], sizes[i], activation))

    def infer(self, a):
        """a (np.array<float>) : an array of floats -- the input to the network"""
        assert len(a) == self.layers[0].input_size()
        a, z = [a], []
        for layer in self.layers:
            z.append((layer.weights @ a[-1]) + layer.biases)
            a.append(layer.act(z[-1]))
        return a, array(z, dtype=object)
