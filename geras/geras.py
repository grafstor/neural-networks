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

    version 3.1:
        - refacktoring

    version 3.2:
        - add comments

    version 3.3:
        - loss foo
           categorical crossentropy
           binary crossentropy
        - add LeakyReLU, Elu, Selu

'''

__version__ = '3.3'

import random
import matplotlib.pyplot as plt
import numpy as np


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
        
    def derivative(self, x):
        return x*(1 - x)


class ReLU:
    def __call__(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)


class Tanh:
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - x * x


class Softmax:
    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def derivative(self, x):
        return x*(1 - x)


class LeakyReLU:
    def __call__(self, x, alpha=0.1):
        return np.where(x >= 0, x, alpha * x)

    def derivative(self, x, alpha=0.1):
        return np.where(x >= 0, 1, alpha)


class Elu:
    def __call__(self, x, alpha=0.1):
        return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1))

    def derivative(self, x, alpha=0.1):
        return np.where(x >= 0.0, 1, self.ELU(x) + alpha)


class Selu:
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def derivative(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))


class Loss:
    def __init__(self):
        self.error = None
        self.delta = None

    def acc_score(self, y, p):
        accuracy = np.sum(y == p) / len(y)
        return accuracy


class Categorical_crossentropy(Loss):

    def __call__(self, is_first_iteration, train_y, lr,
                 previous_layer, present_layer,
                 previous_layer_result, present_layer_result):

        if is_first_iteration:
            self.delta = present_layer_result-train_y

        else:
            self.error = np.dot(self.delta, present_layer.weights.T)
            self.delta = self.error*previous_layer.activation.derivative(present_layer_result)

        weights_change = lr*np.dot(previous_layer_result.T, self.delta)
        bias_change = lr*np.sum(self.delta, axis=0, keepdims=True)

        return (-weights_change, -bias_change)

    def acc(self, y_true, y_pred):
        accuracy = self.acc_score(np.argmax(y_true, axis=1),
                                  np.argmax(y_pred, axis=1))
        return accuracy


class Binary_crossentropy(Loss):

    def __call__(self, is_first_iteration, train_y, lr,
                 previous_layer, present_layer,
                 previous_layer_result, present_layer_result):

        if is_first_iteration:
            self.error = train_y - present_layer_result

        else:
            self.error = np.dot(self.delta, present_layer.weights.T)

        self.delta = self.error*previous_layer.activation.derivative(present_layer_result)

        weights_change = lr*np.dot(previous_layer_result.T, self.delta)
        bias_change = lr*np.mean(self.delta, axis=0)

        return (weights_change, bias_change)

    def acc(self, y_true, y_pred):
        accuracy = self.acc_score(y_true, y_pred)
        return accuracy


class Dense:
    '''
    main sequential layer
    '''
    def __init__(self, neurons:int, activation:str='sigmoid', dropout=0.0):
        '''
        get num of neurons,
            layer activation,
            dropout
        '''

        assert 0 <= dropout <= 1

        self.neurons = neurons
        self.dropout = dropout

        activations = {
                'sigmoid': Sigmoid(),
                'relu': ReLU(),
                'tanh': Tanh(),
                'softmax': Softmax(),
                'leakyrelu': LeakyReLU(),
                'elu': Elu(),
                'selu': Selu(),
            }

        try:
            self.activation = activations[activation]
        except:
            raise MyException('Uncnown activation')


    def activate(self, input_num:int):
        '''
        get shape of previous layer
        connect this layer with previous
        '''
        self.weights = 2*np.random.random((input_num, self.neurons),) - 1
        self.bias = 0

    def feed(self, data:int):
        '''
        get data for layer
        feed forword this layer
        '''
        return self.activation(np.dot(data, self.weights)) + self.bias


class Input:
    '''
    layer used to set the incoming data shape
    '''
    def __init__(self, input_shape:int):
        self.shape = input_shape


class Model:
    '''
    sequential model
    '''
    def __init__(self):
        '''
        declares main variables
        '''
        self.layers = []
        self.history = {}

        self.is_test = False

    def add(self, layer):
        '''
        add layer to model
        '''
        self.layers.append(layer)

    def compile(self, loss='binary'):
        '''
        passes the shape from layer to layer
        choose loss
        '''

        losses = {'binary': Binary_crossentropy(),
                  'categorical': Categorical_crossentropy()}

        try:
            self.loss = losses[loss]
        except:
            raise MyException('Uncnown loss')

        input_layer = self.layers.pop(0)
        last_num = input_layer.shape

        for i in range(len(self.layers)):
            self.layers[i].activate(last_num)
            last_num = self.layers[i].neurons

    def fit(self, train_X:list, train_Y:list,
            epochs:int, learning_rate:float=0.1,
            shuffle:bool=True, validation_split:float=0.0,
            view_stat:bool=True, view_error:bool=True):
        '''
        neural network training loop
        get data and labels,
            number of epochs,
            learning rate - speed of learning,
            shuffle data,
            validation split - size of the test sample,
            view stat,
            view error
        '''

        (train_X, train_Y), (test_X, test_Y) = self.__prepare_data(train_X, train_Y,
                                                                   shuffle=shuffle,
                                                                   validation_split=validation_split)

        self.history = {
                'train': [],
                'test': []
            }

        random_state = np.random.RandomState(123)

        for epoch in range(epochs):

            layers_results = self.__feed_forward(train_X, random_state)

            changes = self.__back_propagation(train_Y, layers_results, learning_rate)

            self.__change_weights(changes)

            if view_error:
                self.__test_results(train_Y, layers_results[-1],
                                    test_X, test_Y,
                                    epoch, epochs)

        if view_stat:
            self.__view_stat()

        if view_error:
            print('')

    def predict(self, layer:list):
        '''
        passes data through a neural network
        return result
        '''
        layer = np.array(layer)
        for i in range(len(self.layers)):
            layer = self.layers[i].feed(layer)
        return layer

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

        return ((X, Y), (test_X, test_Y))

    def __prepare_data(self, X, Y, shuffle, validation_split):

        assert 0 <= validation_split <= 1

        if shuffle:
            X, Y = self.__shuffle_data(X, Y)

        self.is_test = bool(validation_split)

        (train_X, train_Y), (test_X, test_Y) = self.__split_data(X, Y, validation_split)

        train_X = np.array(train_X)

        if type(train_Y[0]) == int:
            train_Y = np.array([train_Y]).T
        else:
            train_Y = np.array(train_Y)

        test_X = np.array(test_X)
        test_Y = np.array(test_Y)

        return (train_X, train_Y), (test_X, test_Y)

    def __feed_forward(self, X, rs):
        layers_results = []

        for i in range(len(self.layers)):
            if i == 0:
                layer_output = X
                layers_results.append(X)

            layer_output = self.layers[i].feed(layer_output)

            if self.layers[i].dropout:
                prop = 1 - self.layers[i].dropout
                mask = rs.binomial(size=layer_output.shape, n=1, p=prop)
                layer_output *= mask/prop

            layers_results.append(layer_output)

        return layers_results

    def __back_propagation(self, train_y, predictions, learning_rate):
        changes = []

        for iteration in reversed(range(len(self.layers))):

            is_first_iteration = iteration == len(self.layers)-1

            previous_layer_result = predictions[iteration]
            present_layer_result = predictions[iteration+1]

            previous_layer = self.layers[iteration]

            if is_first_iteration:
                present_layer = None
            else:
                present_layer = self.layers[iteration+1]

            weights_change, bias_change = self.loss(is_first_iteration,
                                                    train_y,
                                                    learning_rate,

                                                    previous_layer,
                                                    present_layer,

                                                    previous_layer_result,
                                                    present_layer_result)

            changes.append([weights_change, bias_change])

        return changes

    def __change_weights(self, changes):
        changes = changes[::-1] # reverse after back prop

        for i in range(len(changes)):
            self.layers[i].weights += changes[i][0]
            self.layers[i].bias += changes[i][1]

    def __test_results(self, train_Y, train_P, test_X, test_Y, epoch, epochs):


        train_acc = round(self.loss.acc(train_Y, train_P), 4)

        self.history['train'].append(train_acc)

        train_E = self.__mse(train_Y, train_P)
        train_E = round(train_E, 4)

        errors_line = f'Train  [loss: {train_E} acc: {train_acc}]   '

        if self.is_test:
            test_P = self.predict(test_X)

            test_acc = round(self.loss.acc(test_Y, test_P), 4)
            self.history['test'].append(test_acc)

            test_E = self.__mse(test_Y, test_P)
            test_E = round(test_E, 4)

            errors_line += f'Test  [loss: {test_E} acc: {test_acc}]   '

        self.__progress_bar(epoch+1, epochs, errors_line)

    def __mse(self, y_true, y_pred):
        error = np.mean(np.power(y_true - y_pred, 2))
        return error

    def __progress_bar(self, current, total, errors_line):
        percent = float(current) * 100 / total
        arrow   = '▆' * int(percent/100 * 30 - 1)
        spaces  = ' ' * (30 - len(arrow))

        print(f'Train:  {arrow}{spaces} {int(percent)}%   {errors_line}', end='\r')

    def __view_stat(self):
        if self.is_test:
            plt.plot(self.history['test'], label='test')

        plt.plot(self.history['train'], label='train')
        plt.xlabel('epoch')
        plt.ylabel('true answers')
        plt.legend()
        plt.show()


class Tokenizer:
    '''
    used to prepare string data to neural network
    '''
    def __init__(self, max_words:int, text:list):
        self.max_words = max_words
        self.unallow_list = '.,:<>?!'
        self.word_index = self.__fit_on_text(text)

    def text_to_sequence(self, text:list):
        '''
        get list with str lines
        replace each word to id
        return matrix
        '''
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
        train_x = vectorize(new_text, maxlen)

        return train_x

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


def vectorize(sequence:list, maxlen:int):
    '''
    convert matrix to one hot encoding
    '''
    sequence = np.array(sequence)
    vectors = np.zeros((len(sequence), maxlen))

    for i, nums in enumerate(sequence):
        vectors[i, nums] = 1
    return vectors
