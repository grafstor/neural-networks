# test for geras version 2.0

'''
    author: grafstor
    date: 08.06.20
'''

__version__ = "1.0"

from geras import Tokenizer, Model, Input, Dense
from numpy import array


def main():
    train_x = [
        'привет как дела',
        'ок увидимся',
        'лан пока',
        'окей я спать',
        'зватра напиши',
        'шо нада',
        'привет',
        'шо делаешь',
        'потом увидимся',
        'го играть',
        ]

    train_y = [
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
        ]

    tokenizer = Tokenizer(1000, train_x)      
    train_x = tokenizer.text_to_sequence(train_x)


    input_len = len(tokenizer.word_index)

    model = Model()

    model.add(Input(input_len))
    model.add(Dense(32, 'relu', dropout=0.2))
    model.add(Dense(16, 'relu', dropout=0.2))
    model.add(Dense(1, 'sigmoid'))

    model.compile()

    model.fit(train_x,
              train_y,
              epochs=500,
              validation_split=0.2)

    # test_line = ['лан зватра напиши'] # [[0.00530597]]
    test_line = ['привет го играть'] # [[0.99992743]]

    test_line = tokenizer.text_to_sequence(test_line)

    result = model.predict(test_line)
    print(f'result: {result}')


if __name__ == '__main__':
    main()