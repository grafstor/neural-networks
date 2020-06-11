#----------------------------#
# Author: grafstor
# Date: 08.06.20
#----------------------------#
'''
Geras
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

    version 3.0:
        - drop out
        - validation split
        - mse loss
        - log epochs
        - mat plot stat


'''

__version__ = '3.0'

import random
import matplotlib.pyplot as plt
import numpy as np

class Activations:
    def __init__(self):
        self.activations_list = {
            'sigmoid':[self.sigmoid, self.dsigmoid],
            'relu':[self.ReLU, self.dReLU],
            'tanh':[self.tanh, self.dtanh],
        }

    def get(self):
        return self.activations_list

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def dsigmoid(self, x):
        return x*(1 - x)

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1. * (x > 0)

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1. - x * x


class Dense:
    def __init__(self, neurons:int, activation:str='sigmoid', dropout=0.0):

        assert 0 <= dropout <= 1

        self.neurons = neurons
        self.dropout = dropout

        activations = Activations().get()

        try:
            self.activation, self.dactivation = activations[activation]
        except:
            raise MyException('Uncnown activation')


    def activate(self, input_num:int):
        self.weights = 2*np.random.random((input_num, self.neurons)) - 1
        self.bias = 0

    def feed(self, data:int):
        return self.activation(np.dot(data, self.weights)) + self.bias


class Input:
    def __init__(self, input_shape:int):
        self.shape = input_shape
        self.is_test = False


class Model:
    def __init__(self):
        self.layers = []
        self.history = {}

    def __mse(self, y_true, y_pred):
        error = np.mean(np.power(y_true - y_pred, 2))
        return error

    def __shuffle_data(self, X, Y):
        shuffle_data = list(zip(X, Y))
        random.shuffle(shuffle_data)

        X, Y = zip(*shuffle_data)

        X = list(X)
        Y = list(Y)
        return (X, Y)

    def __split_data(self, X, Y, split_size):
        test_data_len = int(len(X)*split_size)

        test_X = []
        test_Y = []

        for i in range(test_data_len):
            random_index = random.randint(0, len(X)-1)

            test_X.append(X.pop(random_index))
            test_Y.append(Y.pop(random_index))

        return ((X, Y),(test_X, test_Y))

    def __prepare_data(self, X, Y, shuffle, validation_split):

        assert 0 <= validation_split <= 1

        if shuffle:
            X, Y = self.__shuffle_data(X, Y)

        self.is_test = bool(validation_split)

        (train_X, train_Y), (test_X, test_Y) = self.__split_data(X, Y, validation_split)

        train_X = np.array(train_X)
        train_Y = np.array([train_Y]).T

        test_X = np.array(test_X)
        test_Y = np.array([test_Y]).T

        return (train_X, train_Y), (test_X, test_Y)

    def add(self, layer):

        self.layers.append(layer)

    def compile(self):
        input_layer = self.layers.pop(0)
        last_num = input_layer.shape

        for i in range(len(self.layers)):
            self.layers[i].activate(last_num)
            last_num = self.layers[i].neurons

    def fit(self, train_X:list, train_Y:list,
            epochs:int, learning_rate:float=0.1,
            shuffle:bool=True, validation_split:float=0.0):

        (train_X, train_Y), (test_X, test_Y) = self.__prepare_data(train_X, train_Y,
                                               shuffle=shuffle,
                                               validation_split=validation_split)

        history = {'train': [],
                   'test': []}

        rs = np.random.RandomState(123)

        for epoch in range(epochs):

            layers_results = [train_X]

            # feed forword
            for i in range(len(self.layers)):
                if i == 0:
                    layer_output = train_X

                layer_output = self.layers[i].feed(layer_output)

                if self.layers[i].dropout:
                    prop = 1-self.layers[i].dropout
                    mask = rs.binomial(size=layer_output.shape,
                                       n=1,
                                       p=prop)
                    layer_output *= mask/prop

                layers_results.append(layer_output)

            changes = []

            # back propagation
            for i in reversed(range(len(self.layers))):

                if i == len(self.layers)-1:
                    error = train_Y - layers_results[i+1]

                else:
                    error = np.dot(delta, self.layers[i+1].weights.T)

                delta = error*self.layers[i].dactivation(layers_results[i+1])

                weights_change = learning_rate*np.dot(layers_results[i].T, delta)
                bias_change = learning_rate*np.mean(delta, axis=0)

                changes.append([weights_change, bias_change])

            # change weights
            for i in range(len(changes)):
                reversed_iteration = len(changes)-i-1

                self.layers[i].weights += changes[reversed_iteration][0]
                self.layers[i].bias += changes[reversed_iteration][1]

            # test data
            train_P = layers_results[-1]

            if self.is_test: test_P = self.predict(test_X)
            else: test_P = 0

            train_E, test_E = self.__test_results(train_Y, train_P,
                                                  test_Y, test_P)

            history['train'].append(train_E)
            history['test'].append(test_E)

            errors = f'Train-Loss: {train_E} ' + \
                    (int(self.is_test)*f'Test-Loss: {test_E}')

            self.__progress(epoch+1, epochs, errors)

        self.__view_stat(history)

        print('')

    def __test_results(self, train_Y, train_P,
                             test_Y, test_P):

        train_E = self.__mse(train_Y, train_P)

        if self.is_test: test_E = self.__mse(test_Y, test_P)
        else: test_E = 0

        return (train_E, test_E)

    def __progress(self, current, total, errors):

        percent = float(current) * 100 / total
        arrow   = '-' * int(percent/100 * 30 - 1) + '>'
        spaces  = ' ' * (30 - len(arrow))

        print(f'Train: [{arrow}{spaces}] {int(percent)}% {errors}', end='\r')

    def predict(self, layer:list):
        layer = np.array(layer)
        for i in range(len(self.layers)):
            layer = self.layers[i].feed(layer)

        layer = [i[0] for i in layer]
        return layer

    def __view_stat(self, history):
        if self.is_test:
            plt.plot(history['test'], label='TEST')

        plt.plot(history['train'], label='TRAIN')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Доля верных ответов')
        plt.legend()
        plt.show()


class Tokenizer:
    def __init__(self, max_words:int, text:list):
        self.max_words = max_words
        self.unallow_list = '.,:<>?!'
        self.word_index = self.__fit_on_text(text)

    def __fit_on_text(self, text:list):
        word_index = {'uncnown': 0}

        for line in text:
            for char in self.unallow_list:
                line = line.replace(char, ' ')

            line = line.split()

            for word in line:
                if word in word_index:
                    word_index[word] += 1
                else:
                    if len(word_index) > self.max_words:
                        word_index['uncnown'] += 1
                    else:
                        word_index[word] = 1

        replace_list = [[word_index[i],i] for i in word_index]
        replace_list.sort(key=lambda a: a[0])
        replace_list = replace_list[::-1]

        word_index = dict((replace_list[i][1],i) for i in range(len(replace_list)))

        return word_index

    def text_to_sequence(self, text:list):
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

        maxlen = len(self.word_index)
        train_x = self.__vectorize(new_text, maxlen)

        return train_x


    def __vectorize(self, sequence:list, maxlen:int):
        sequence = np.array(sequence)
        vectors = np.zeros((len(sequence), maxlen))

        for i, nums in enumerate(sequence):
            vectors[i, nums] = 1
        return vectors
