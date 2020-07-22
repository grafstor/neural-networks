#-------------------------------#
#       Author: grafstor        
#       Date: 12.06.20          
#-------------------------------#

from geras import *

import pandas as pd

import matplotlib.pyplot as plt

def vectorize(sequence:list, maxlen:int):
    sequence = np.array(sequence)
    vectors = np.zeros((len(sequence), maxlen))

    for i, nums in enumerate(sequence):
        vectors[i, nums] = 1
    return vectors

def load_data(main_path):
    train = pd.read_csv(main_path+"/train.csv")
    test = pd.read_csv(main_path+"/test.csv")

    train_x = train.drop(labels=["label"], axis=1)
    train_y = train["label"]

    train_x /= 255
    test /= 255

    train_x = np.array(train_x[:1000])
    train_y = np.array(train_y[:1000])
    test = np.array(test)

    train_y = vectorize(train_y, 10)

    return ((train_x, train_y), (test))

def main():
    data_path = 'test train data/mnist'

    (train_x, train_y), (test) = load_data(data_path)

    lr = 0.001

    model = Model(

        Dense(784, lr),
        Sigmoid(),
        Dropout(0.5),

        Dense(10, lr),
        Softmax(),
        
    )

    model.fit(train_x, train_y, 500)

    test_case = np.array([test[5]])
    test_result = model.predict(test_case)[0]

    plt.bar([str(i) for i in range(0,10)], result)
    plt.xlabel("Numbers")
    plt.ylabel("Probability")
    plt.show()


if __name__ == '__main__':
    main()