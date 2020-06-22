# test for geras version 3.0

'''
    author: grafstor
    date: 12.06.20
'''

__version__ = "1.0"

from geras import Model, Input, Dense, vectorize

from numpy import array
import pandas as pd

import matplotlib.pyplot as plt

def load_data(main_path):
    train = pd.read_csv(main_path+"/train.csv")
    test = pd.read_csv(main_path+"/test.csv")

    train_x = train.drop(labels=["label"], axis=1)
    train_y = train["label"]

    train_x /= 255
    test /= 255

    train_x = array(train_x[:2000])
    train_y = array(train_y[:2000])
    test = array(test[:2000])

    train_y = vectorize(train_y, 10)

    return ((train_x, train_y), (test))

def main():
    data_path = 'test train data/mnist'

    (train_x, train_y), (test) = load_data(data_path)

    model = Model()

    model.add(Input(784))
    model.add(Dense(784, 'relu', dropout=0.5))
    model.add(Dense(10, 'softmax'))

    model.compile(loss='categorical')

    model.fit(train_x,
              train_y,
              epochs=100,
              validation_split=0.2,
              learning_rate=0.0001,
              )

    num = 5

    result = model.predict(test[num:num+1])
    fruits = [str(i) for i in range(0,10)]

    plt.bar(fruits, result[0])
    plt.xlabel("Цифры")
    plt.ylabel("Вероятность")
    plt.show()

if __name__ == '__main__':
    main()