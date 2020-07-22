#-------------------------------#
#       Author: grafstor        
#       Date: 08.06.20          
#-------------------------------#

'''
module to build simple neural network
'''

import numpy as np

class Model:
    def __init__(self, *layers):
        self.layers = layers
        self.loss = Crossentropy()

    def fit(self, x, y, epochs):
        for epoch in range(epochs):
            prediction = self.__feedforward(x)
            gradient = self.loss(y, prediction)
            self.__backpropogation(gradient)

    def predict(self, x):
        return self.__feedforward(x)

    def __feedforward(self, x):
        result = x
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def __backpropogation(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


class Layer:
    def __init__(self):
        self.neurons = None
        self.last_data = None

    def forward(self, data):
        pass

    def backward(self, gradient):
        pass


class Dense(Layer):
    def __init__(self, neurons, learning_rate=1):
        self.neurons = neurons
        self.learning_rate = learning_rate

        self.weights = []
        self.bias = []

    def forward(self, data):
        if not len(self.weights):
            self.weights = 2*np.random.random((data.shape[1], self.neurons)) - 1
            self.bias = np.zeros((1, self.neurons))

        self.last_data = data

        result = np.dot(data, self.weights) + self.bias
        return result

    def backward(self, gradient):
        next_gradient = np.dot(gradient, self.weights.T)

        self.weights += self.learning_rate * np.dot(self.last_data.T, gradient)
        self.bias += self.learning_rate * np.sum(gradient, axis=0, keepdims=True)

        return next_gradient


class Dropout(Layer):
    def __init__(self, dropout):
        self.dropout = dropout

    def forward(self, data):
        probability = 1.0 - self.dropout

        mask = np.random.binomial(size=data.shape, n=1, p=probability)
        data *= mask/probability
        return data

    def backward(self, gradient):
        return gradient


class Activation(Layer):
    def forward(self, data):
        data = self(data)
        self.last_data = data
        return data

    def backward(self, gradient):
        return gradient * self.derivative(self.last_data)

    def derivative(self, x):
        pass


class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x*(1 - x)


class Softmax(Activation):
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, x):
        return x


class ReLU(Activation):
    def __call__(self, x):
        self.last_input_data = x
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)


class Loss:
    def __init__(self):
        pass

    def acc_score(self, y, p):
        accuracy = np.sum(y == p) / len(y)
        return accuracy


class Crossentropy(Loss):
    def __call__(self, y, p):
        return (y - p)

    def acc(self, y, p):
        return self.acc_score(np.argmax(y, axis=1), np.argmax(p, axis=1))
