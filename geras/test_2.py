# test for geras version 2.0

'''
    author: grafstor
    date: 08.06.20
'''

__version__ = "1.0"

from geras import Tokenizer, vectorize, Model, Input, Dense
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

    train_y = [[
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
        ]]

    tokenizer = Tokenizer()        
    tokenizer.fit_on_text(train_x)

    train_x = tokenizer.text_to_sequence(train_x)

    train_x = array(train_x)
    train_y = array(train_y).T

    maxlen = len(tokenizer.word_index)
    train_x = vectorize(train_x, maxlen)


    model = Model()

    model.add(Input(maxlen))
    model.add(Dense(32, 'relu'))
    model.add(Dense(16, 'relu'))
    model.add(Dense(1, 'sigmoid'))

    model.compile()

    model.fit(train_x,
              train_y,
              epochs=1000)


    # test_line = ['лан зватра напиши'] # [[0.00530597]]
    test_line = ['привет го играть'] # [[0.99992743]]

    test_line = tokenizer.text_to_sequence(test_line)
    test_line = vectorize(test_line, maxlen)

    result = model.predict(test_line)
    print(f'result: {result}')


if __name__ == '__main__':
    main()