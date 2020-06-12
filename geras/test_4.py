# test for geras version 3.0

'''
    author: grafstor
    date: 12.06.20
'''

__version__ = "1.0"

from geras import Model, Input, Dense
from numpy import array
from PIL import Image, ImageTk
import tkinter
import time

x_list = []
y_list = []

height = 190
width = 190

def callback_1(event):
    global x_list, y_list

    x_list.append([event.y, event.x])
    y_list.append(1)

    lol(x_list, y_list)

def callback_2(event):
    global x_list, y_list

    x_list.append([event.y, event.x])
    y_list.append(0)

    lol(x_list, y_list)

def lol(train_x, train_Y):

    photoshop = Image.new("RGB", (height, width))
    pixels = photoshop.load()

    train_X = array(train_x)/height

    model = Model()

    model.add(Input(2))
    model.add(Dense(100, 'sigmoid'))
    model.add(Dense(80, 'sigmoid'))
    model.add(Dense(1, 'sigmoid'))

    model.compile()

    model.fit(train_X,
              train_Y,
              epochs=1000 + len(train_x)*190,
              view_stat=False)

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

    for i,(x,y) in enumerate(train_x):
        pixels[int(y),int(x)] = ((1-train_Y[i])*255,
                                 (1-train_Y[i])*255,
                                 (1-train_Y[i])*255)

    img_photo = ImageTk.PhotoImage(photoshop)

    label["image"] = img_photo
    label.image = img_photo

if __name__ == '__main__':

    win = tkinter.Tk()

    win.geometry(f'{width}x{height}')

    win.bind("<Button-1>", callback_1)
    win.bind("<Button-3>", callback_2)

    label = tkinter.Label(win)
    label.pack()

    win.mainloop()