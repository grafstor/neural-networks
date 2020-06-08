# geras

'''
    author: grafstor
    date: 08.06.20

    version 1.0:
        - Activation
            - sigmoid
            - sigmoid_deviv
            - ReLU
            - dReLU
        - Model class
            - add
            - compile
            - fit
            - predict
        - Dense class
            - activate
            - feed

    exaple:
        >>> nn = Model()
        ... nn.add(Dense(10, 'relu'))
        ... nn.add(Dense(30, 'relu'))
        ... nn.add(Dense(1, 'sigmoid'))
        >>> nn.fit(x, y, epochs=1000)
        >>> nn.predict(test)
        0.9939795696

'''

__version__ = '1.0'

import numpy as np
import random

class Activation:

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_deviv(x):
        return x*(1 - x)

    def ReLU(x):
        return x * (x > 0)

    def dReLU(x):
        return 1. * (x > 0)

class Dense:
    def __init__(self, neurons, activation='sigmoid'):
        self.neurons = neurons

        if activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.dactivation = Activation.sigmoid_deviv

        elif activation == 'relu':
            self.activation = Activation.ReLU
            self.dactivation = Activation.dReLU

        else:
            raise MyException('Uncnown activation')

    def activate(self, input_num):
        self.syn = 2*np.random.random((input_num, self.neurons)) - 1

    def feed(self, data):
        return self.activation(np.dot(data, self.syn))

class Input:
    def __init__(self, input_shape):
        self.shape = input_shape


class Model:
    def __init__(self):
        self.layers = []

    def add(self,layer):
        self.layers.append(layer)

    def compile(self):
        input_layer = self.layers.pop(0)
        last_num = input_layer.shape 

        for i in range(len(self.layers)):
            self.layers[i].activate(last_num)
            last_num = self.layers[i].neurons


    def fit(self, train_x, train_y, epochs):

        train_x = np.array(train_x)
        train_y = np.array(train_y).T


        for _ in range(epochs):
            layers = []
            changes = []
            layer = train_x

            for i in range(len(self.layers)):
                layers.append(layer)
                layer = self.layers[i].feed(layer)


            error = (train_y - layer)
            delta = error*self.layers[-1].dactivation(layer)
            changes.append(np.dot(layers[-1].T, delta))


            for i in range(len(layers)-1, 0, -1):
                error = np.dot(delta, self.layers[i].syn.T)
                delta = error*self.layers[i-1].dactivation(layers[i])     
                changes.append(np.dot(layers[i-1].T, delta))

            for i in range(len(changes)):
                self.layers[i].syn += changes[len(changes)-i-1]

    def predict(self, data):
        layer = np.array(data)
        for i in range(len(self.layers)):
            layer = self.layers[i].feed(layer)
        return layer
