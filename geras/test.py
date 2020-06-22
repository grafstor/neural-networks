# test for geras version 1.0

'''
    author: grafstor
    date: 08.06.20
'''

__version__ = "2.0"

from geras import Input, Dense, Model

def main():
    x = [[1,0,1],
         [0,0,1],
         [1,1,0],
         [0,1,0]]

    y = [1,0,1,0]

    model = Model()

    model.add(Input(3))
    model.add(Dense(3, 'sigmoid'))
    model.add(Dense(1, 'sigmoid'))

    model.compile()

    model.fit(x, y, epochs=5000)


    test = [[1,0,1]]

    result = model.predict(test)
    print(result[0])

if __name__ == '__main__':
    main()