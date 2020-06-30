#----------------------------#
# Author: grafstor
# Date: 30.06.20
#----------------------------#

__version__ = '0.x'

import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        self.loss = Crossentropy()

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, epochs):
        x, y = prepare(x, y)

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
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation

        self.weights = []
        self.bias = []

    def forward(self, data):
        if not len(self.weights):
            self.weights = 2*np.random.random((data.shape[1], self.neurons)) - 1
            self.bias = np.zeros((1, self.neurons))

        self.last_data = data

        result = np.dot(data, self.weights) + self.bias
        result = self.activation.forward(result)

        return result

    def backward(self, gradient):
        gradient = self.activation.backward(gradient)

        next_gradient = np.dot(gradient, self.weights.T)

        self.weights += np.dot(self.last_data.T, gradient)
        self.bias += np.sum(gradient, axis=0, keepdims=True)

        return next_gradient


class Activation(Layer):
    def forward(self, data):
        data = self(data)
        self.last_data = data
        return data

    def backward(self, gradient):
        return gradient * self.derivative(self.last_data)


class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x*(1 - x)


class Loss:
    def __init__(self):
        pass


class Crossentropy(Loss):
    def __call__(self, y, y_pred):
        return (y - y_pred)


def prepare(x, y):
    x = np.array(x, dtype=np.float128)
    y = np.array([y], dtype=np.float128).T
    return (x, y)


def complete_test():
    x = [[1,0,1],
         [0,0,1],
         [1,1,0],
         [0,1,0]]

    y = [1,0,1,0]

    model = Model()

    # model.add(Dense(3, Sigmoid()))

    # model.add(Dense(1, Sigmoid()))

    model.layers = [Dense(3, Sigmoid()),
                    Dense(1, Sigmoid())]

    model.fit(x, y, 10000)


    print(model.predict([1,0,1]))
    print(model.predict([0,1,0]))

if __name__ == '__main__':
    complete_test()
