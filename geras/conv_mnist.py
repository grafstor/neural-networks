#-------------------------------#
#       Author: grafstor        
#       Date: 10.08.20          
#-------------------------------#

from geras import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def vectorize(sequence:list, maxlen:int):
    sequence = np.array(sequence)
    vectors = np.zeros((len(sequence), maxlen))

    for i, nums in enumerate(sequence):
        vectors[i, nums] = 1
    return vectors

def load_data(main_path):
    train = pd.read_csv(main_path+"/train.csv")

    train_x = train.drop(labels=["label"], axis=1)
    train_y = train["label"]

    train_x /= 255

    border = len(train_x)//10

    test_x = np.array(train_x[:border])
    test_y = np.array(train_y[:border])

    train_x = np.array(train_x[border:])
    train_y = np.array(train_y[border:])

    test_x = test_x.reshape(-1, 1, 28, 28)
    train_x = train_x.reshape(-1, 1, 28, 28)

    train_y = vectorize(train_y, 10)
    test_y = vectorize(test_y, 10)

    return ((train_x, train_y), (test_x, test_y))

def main():
    data_path = 'test train data/mnist'

    (train_x, train_y), (test_x, test_y) = load_data(data_path)


    model = Model(

        Conv2D(55, (5,5)),
        ReLU(),

        Flatten(),

        Dense(512),
        LeakyReLU(),

        Dropout(0.5),

        Dense(10),
        Softmax(),
    
    )(Adam(0.001))

    bc = 512

    print(len(train_x)//bc+1)

    for epoch in range(5):
        for i in range(len(train_x)//bc+1):
            train_X = train_x[i*bc:(i+1)*bc]
            train_Y = train_y[i*bc:(i+1)*bc]

            prediction = model.train(train_X, train_Y)

            acc = model.loss.acc(train_Y, prediction)

            print(i, acc)

    with open('conv_mnist_model.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
