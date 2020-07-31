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

    train_x = train.drop(labels=["label"], axis=1)
    train_y = train["label"]

    train_x /= 255

    border = len(train_x)//10

    test_x = np.array(train_x[:border])
    test_y = np.array(train_y[:border])

    train_x = np.array(train_x[border:])
    train_y = np.array(train_y[border:])

    train_y = vectorize(train_y, 10)
    test_y = vectorize(test_y, 10)

    return ((train_x, train_y), (test_x, test_y))

def main():
    data_path = 'test train data/mnist'

    (train_x, train_y), (test_x, test_y) = load_data(data_path)


    model = Model(

        Dense(783),
        Sigmoid(),

        Dropout(0.5),

        Dense(10),
        Softmax(),
    
    )(Adam(0.001))

    bc = 256

    for epoch in range(3):
        print('Epoch', epoch, 'training..', end='\r')

        for i in range(len(train_x)//bc+1):
            train_X = train_x[i*bc:(i+1)*bc]
            train_Y = train_y[i*bc:(i+1)*bc]

            prediction = model.train(train_X, train_Y)

        print('Epoch', epoch, 'done      ')

        if epoch%1 == 0:
            prediction = model.predict(train_x) 
            acc = model.loss.acc(train_y, prediction)

            test_prediction = model.predict(test_x) 
            test_acc = model.loss.acc(test_y, test_prediction)

            print('train_acc', acc, ' test_acc', test_acc)

    test_case = np.array([test_x[1]])
    test_result = model.predict(test_case)[0]

    plt.bar([str(i) for i in range(0,10)], test_result)
    plt.xlabel("Numbers")
    plt.ylabel("Probability")
    plt.show()


if __name__ == '__main__':
    main()