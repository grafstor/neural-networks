# geras

'''
    author: grafstor
    date: 08.06.20

    version 1.0:
        - Activation
            - sigmoid
            - dsigmoid
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

    version 2.0:
        - Tokenizer
            - fit_on_text
            - text_to_sequence
        - vectorize

    version 2.1:
        - Activation
            - tanh
            - dtanh

    version 2.2:
        - shuffle data
        - bias
        - learning_rate

    exaple:
        >>> nn = Model()
        ... nn.add(Dense(10, 'relu'))
        ... nn.add(Dense(30, 'relu'))
        ... nn.add(Dense(1, 'sigmoid'))
        >>> nn.fit(x, y, epochs=1000)
        >>> nn.predict(test)
        0.9939795696

'''

__version__ = '2.2'

import numpy as np
import random

class Activation:

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    def dsigmoid(x):
        return x*(1 - x)

    def ReLU(x):
        return x * (x > 0)

    def dReLU(x):
        return 1. * (x > 0)

    def tanh(x):
        return np.tanh(x)

    def dtanh(x):
        return 1. - x * x


class Dense:
    def __init__(self, neurons, activation='sigmoid'):
        self.neurons = neurons

        if activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.dactivation = Activation.dsigmoid

        elif activation == 'relu':
            self.activation = Activation.ReLU
            self.dactivation = Activation.dReLU

        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.dactivation = Activation.dtanh

        else:
            raise MyException('Uncnown activation')

    def activate(self, input_num):
        self.syn = 2*np.random.random((input_num, self.neurons)) - 1
        self.bias = 0

    def feed(self, data):
        return self.activation(np.dot(data, self.syn)) + self.bias


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


    def fit(self, train_x, train_y, epochs, shuffle=True, learning_rate=0.1):

        if shuffle:
            shuffle_data = list(zip(train_x, train_y))
            random.shuffle(shuffle_data)
            train_x, train_y = zip(*shuffle_data)
            del shuffle_data

            train_x = np.array(train_x)
            train_y = np.array(train_y)

        for _ in range(epochs):

            layers = [train_x]

            changes = []
            changes_bias = []

            # feed forword
            for i in range(len(self.layers)):
                if i == 0:
                    layer_output = train_x
                layer_output = self.layers[i].feed(layer_output)
                layers.append(layer_output)

            # back propagation
            for i in reversed(range(1, len(layers))):
                if i == len(layers)-1:
                    error = train_y - layers[i]
                else:
                    error = np.dot(delta, self.layers[i].syn.T)
                delta = error*self.layers[i-1].dactivation(layers[i]) 

                changes.append(learning_rate*np.dot(layers[i-1].T, delta))
                changes_bias.append(learning_rate*np.mean(delta, axis=0))

            # change weights
            for i in range(len(changes)):
                self.layers[i].syn += changes[len(changes)-i-1]
                self.layers[i].bias += changes_bias[len(changes)-i-1]

    def predict(self, data):
        layer = np.array(data)
        for i in range(len(self.layers)):
            layer = self.layers[i].feed(layer)
        return layer


class Tokenizer:
    def __init__(self, max_words=5000):
        self.max_words = max_words
        self.word_index = {'uncnown': 0}
        self.unallow_list = '.,:<>?!'

    def fit_on_text(self, text):
        for line in text:
            for char in self.unallow_list:
                line = line.replace(char, ' ')

            line = line.split()

            for word in line:
                if word in self.word_index:
                    self.word_index[word] += 1
                else:
                    if len(self.word_index) > self.max_words:
                        self.word_index['uncnown'] += 1
                    else:
                        self.word_index[word] = 1

        replace_list = [[self.word_index[i],i] for i in self.word_index]
        replace_list.sort(key=lambda a: a[0])
        replace_list = replace_list[::-1]

        self.word_index = dict((replace_list[i][1],i) for i in range(len(replace_list)))

    def text_to_sequence(self, text):
        new_text = []

        for line in text:
            new_line = []

            for char in self.unallow_list:
                line = line.replace(char, ' ')

            line = line.split()
            for word in line:
                try:
                    new_line.append(self.word_index[word])
                except:
                    new_line.append(self.word_index['uncnown'])

            new_text.append(new_line)

        return new_text


def vectorize(seq, veckor=100):
    vectors = np.zeros((len(seq), veckor))
    for i, nums in enumerate(seq):
        vectors[i, nums] = 1
    return vectors
