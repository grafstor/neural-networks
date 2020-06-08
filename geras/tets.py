# test for geras version 1.0

'''
    author: grafstor
    date: 08.06.20
'''

__version__ = "1.0"

from geras import Input, Dense, Model

def main():
    x = [[1,0,1],
         [0,0,1],
         [1,1,0],
         [0,1,0]]

    y = [[1,
    	  0,
    	  1,
    	  0]]

    nn = Model()

    nn.add(Input(3))
    nn.add(Dense(30, 'relu'))
    nn.add(Dense(15, 'relu'))
    nn.add(Dense(1, 'sigmoid'))

    nn.compile()

    nn.fit(x, y, epochs=1000)


    test = [[1,0,1]]
    result = nn.predict(test)
    print(result[0][0])

if __name__ == '__main__':
    main()