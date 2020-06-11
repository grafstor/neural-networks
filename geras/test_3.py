# test for geras version 2.2

'''
    author: grafstor
    date: 08.06.20
'''

__version__ = "1.0"

from geras import Input, Dense, Model
from numpy import array
from PIL import Image


def main():
    height = 300
    width = 300

    photoshop = Image.new("RGB", (height, width))
    pixels = photoshop.load()

    train_x = [[50, 70],[50,30],[30,50],
                [40, 10],[30, 30],[40, 80],
                [20,30],[30,40],

                [40, 50], [60,30], [60,70],
                [70, 20],[80, 30],[70, 80],
                [70,50], [50, 50]]

    train_y = [1,1,1,
                1,1,1,
                1,1,
                
                0,0,0,
                0,0,0,
                0,0]

    train_x_to_dot = array(train_x)*3

    train_x = array(train_x_to_dot)/height

    model = Model()

    model.add(Input(2))
    model.add(Dense(128, 'sigmoid'))
    model.add(Dense(100, 'sigmoid'))
    model.add(Dense(1, 'sigmoid'))

    model.compile()


    model.fit(train_x,
              train_y,
              epochs=15000)

    test = []
    for x in range(width):
        for y in range(height):
            test.append([y,x])
    result = model.predict(array(test)/height)
    result = array(result)
    result = result.reshape(width, height,1)

    for x in range(width):
        for y in range(height):
            dot = result[y,x]
            
            pixels[y, x] = (int(dot*255),
                           int(dot*255),
                           int(dot*255))

    for i,(x,y) in enumerate(train_x_to_dot):
        pixels[int(y),int(x)] = ((1-train_y[i])*255,
                                (1-train_y[i])*255,
                                (1-train_y[i])*255)
    photoshop.save('test.png', "PNG")


if __name__ == '__main__':
    main()