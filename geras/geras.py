#-------------------------------#
#       Author: grafstor        
#       Date: 08.06.20          
#-------------------------------#

'''
module to build simple neural network
'''

import numpy as np
import copy
import math

class Model:
    def __init__(self, *layers):
        self.layers = layers
        self.loss = Crossentropy()

    def __call__(self, optimizer, loss=None):
        self.loss = loss if loss else self.loss
        for layer in self.layers:
            layer(optimizer)
        return self

    def train(self, x, y):
        prediction = self.__feedforward(x)
        gradient = self.loss(y, prediction)
        self.__backpropogation(gradient)
        return prediction

    def predict(self, x):
        return self.__feedforward(x)

    def __feedforward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def __backpropogation(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


class Layer:
    def __init__(self):
        self.neurons = None
        self.last_data = None
        self.trainable = True

    def __call__(self, optimizer):
        pass

    def forward(self, data):
        pass

    def backward(self, gradient):
        pass


class Dense(Layer):
    def __init__(self, neurons, trainable=True):
        self.neurons = neurons
        self.trainable = trainable

        self.weights = []
        self.bias = []

    def __call__(self, optimizer):
        self.weights_optimizer = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)

    def forward(self, data):
        if not len(self.weights):
            limit = 1 / math.sqrt(data.shape[1])
            self.weights  = np.random.uniform(-limit, limit, (data.shape[1], self.neurons))
            self.bias = np.zeros((1, self.neurons))

        self.last_data = data

        output = np.dot(data, self.weights) + self.bias
        return output

    def backward(self, gradient):
        next_gradient = np.dot(gradient, self.weights.T)

        if self.trainable:
            weights_gradient = np.dot(self.last_data.T, gradient)
            bias_gradient = np.sum(gradient, axis=0, keepdims=True)

            self.weights -= self.weights_optimizer.update(weights_gradient)
            self.bias -= self.bias_optimizer.update(bias_gradient)
        
        return next_gradient


class Dropout(Layer):
    def __init__(self, dropout):
        self.dropout = dropout

    def forward(self, data):
        probability = 1.0 - self.dropout

        self.mask = np.random.binomial(size=data.shape, n=1, p=probability)
        data *= self.mask/probability
        return data

    def backward(self, gradient):
        return gradient * self.mask


class Reshape(Layer):
    def __init__(self, shape):
        self.shape = shape
        self.previous_shape = None

    def forward(self, data):
        self.previous_shape = data.shape
        return data.reshape((-1, *self.shape))

    def backward(self, gradient):
        return gradient.reshape(self.previous_shape)


class Flatten(Layer):
    def __init__(self):
        self.previous_shape = None

    def forward(self, data):
        self.previous_shape = data.shape
        return data.reshape((data.shape[0], -1))

    def backward(self, gradient):
        return gradient.reshape(self.previous_shape)


class Activation(Layer):
    def forward(self, data):
        data = self.forward_pass(data)
        self.last_data = data
        return data

    def backward(self, gradient):
        return gradient * self.derivative(self.last_data)

    def derivative(self, x):
        pass


class Sigmoid(Activation):
    def forward_pass(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class Softmax(Activation):
    def forward_pass(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, x):
        return x * (1 - x)


class ReLU(Activation):
    def forward_pass(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)


class TanH(Activation):
    def forward_pass(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.power(x, 2)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward_pass(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x >= 0, 1, self.alpha)


class Loss:
    def __init__(self):
        pass

    def acc_score(self, y, p):
        accuracy = np.sum(y == p) / len(y)
        return accuracy


class Crossentropy(Loss): 
    def __call__(self, y, p):
        return (p - y)/(p * (1 - p))

    def loss(self, y, p):
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return self.acc_score(np.argmax(y, axis=1), np.argmax(p, axis=1))


class Optimizer:
    def __init__(self):
        self.learning_rate = None

    def update(self):
        pass


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8

        self.m = None
        self.v = None

        self.b1 = beta_1
        self.b2 = beta_2

    def update(self, gradient):
        if self.m is None:
            self.m = np.zeros(np.shape(gradient))
            self.v = np.zeros(np.shape(gradient))
        
        self.m = self.b1 * self.m + (1 - self.b1) * gradient
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(gradient, 2)

        m_deriv = self.m / (1 - self.b1)
        v_deriv = self.v / (1 - self.b2)

        weights_update = self.learning_rate * m_deriv / (np.sqrt(v_deriv) + self.epsilon)

        return  weights_update


class BatchNormalization(Layer):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.trainable = True
        self.epsilon = 0.01

        self.running_mean = None
        self.running_var = None

        self.input_shape = None 

    def __call__(self, optimizer):
        self.gamma_optimizer  = copy.copy(optimizer)
        self.beta_optimizer = copy.copy(optimizer)

    def forward(self, data, training=True):

        if self.running_mean is None:
            self.input_shape = data.shape[1:]

            self.gamma  = np.ones(self.input_shape)
            self.beta = np.zeros(self.input_shape)

            self.running_mean = np.mean(data, axis=0)
            self.running_var = np.var(data, axis=0)

        if training and self.trainable:
            mean = np.mean(data, axis=0)
            var = np.var(data, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
        else:
            mean = self.running_mean
            var = self.running_var

        self.data_centered = data - mean
        self.stddev_inv = 1 / np.sqrt(var + self.epsilon)

        data_norm = self.data_centered * self.stddev_inv
        output = self.gamma * data_norm + self.beta

        return output

    def backward(self, gradient):

        batch_size = gradient.shape[0]

        next_gradient = (1 / batch_size) * self.gamma * self.stddev_inv * (batch_size * gradient
            -np.sum(gradient, axis=0) - self.data_centered * self.stddev_inv**2 * np.sum(gradient * self.data_centered, axis=0))

        if self.trainable:
            data_norm = self.data_centered * self.stddev_inv

            gamma_gradient = np.sum(gradient * data_norm, axis=0)
            beta_gradient = np.sum(gradient, axis=0)

            self.gamma -= self.gamma_optimizer.update(gamma_gradient)
            self.beta -= self.beta_optimizer.update(beta_gradient)

        return next_gradient


class Conv2D(Layer):
    def __init__(self, n_filters, filter_shape):
        self.n_filters = n_filters
        self.filter_shape = filter_shape

        self.trainable = True

        self.weights = None
        self.bias = None

    def __call__(self, optimizer):

        self.weights_optimizer = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)

    def forward(self, data):
        if self.weights is None:

            self.input_shape = data.shape[1:]

            filter_height, filter_width = self.filter_shape

            channels = self.input_shape[0]
            limit = 1 / math.sqrt(np.prod(self.filter_shape))

            self.weights = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width))
            self.bias = np.zeros((self.n_filters, 1))

        self.layer_input = data

        batch_size, channels, height, width = data.shape

        self.data_col = image_to_column(data, self.filter_shape)
        self.W_col = self.weights.reshape((self.n_filters, -1))

        output = self.W_col.dot(self.data_col) + self.bias
        output = output.reshape(self.output_shape() + (batch_size, ))

        return output.transpose(3,0,1,2)

    def backward(self, gradient):
        gradient = gradient.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        if self.trainable:
            grad_weights = gradient.dot(self.data_col.T).reshape(self.weights.shape)
            grad_bias = np.sum(gradient, axis=1, keepdims=True)

            self.weights -= self.weights_optimizer.update(grad_weights)
            self.bias -= self.bias_optimizer.update(grad_bias)

        gradient = self.W_col.T.dot(gradient)
        gradient = column_to_image(gradient,
                                    self.layer_input.shape,
                                    self.filter_shape)
        return gradient

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape)

        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / 2
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / 2

        return self.n_filters, int(output_height), int(output_width)


def image_to_column(images, filter_shape):
    channels = images.shape[1]

    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape)

    images_padded  = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w))

    cols = images_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols

def column_to_image(cols, images_shape, filter_shape):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape)

    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)

    images_padded = np.zeros((batch_size, channels, height_padded, width_padded))
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w))

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)

    np.add.at(images_padded, (slice(None), k, i, j), cols)
    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]


def determine_padding(filter_shape):

    filter_height, filter_width = filter_shape

    pad_h1 = int(math.floor((filter_height - 1)/2))
    pad_h2 = int(math.ceil((filter_height - 1)/2))
    pad_w1 = int(math.floor((filter_width - 1)/2))
    pad_w2 = int(math.ceil((filter_width - 1)/2))

    return (pad_h1, pad_h2), (pad_w1, pad_w2)

def get_im2col_indices(images_shape, filter_shape, padding):
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding

    out_height = int((height + np.sum(pad_h) - filter_height) / 2)
    out_width = int((width + np.sum(pad_w) - filter_width) / 2)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    return (k, i, j)
