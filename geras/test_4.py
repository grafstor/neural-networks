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

model = 0

height = 100
width = 100

def callback_1(event):
    global x_list, y_list
    x_list.append([event.y, event.x])
    y_list.append(1)

    new_model(x_list, y_list)

def callback_3(event):
    global x_list, y_list
    x_list.append([event.y, event.x])
    y_list.append(0)

    new_model(x_list, y_list)

def callback_2(event):
    train_X = array(x_list)/height

    model.fit(train_X,
              y_list,
              epochs=10000 + len(x_list)*190,
              view_stat=False)
    upload_img()

def new_model(train_x, train_Y):
    global model

    train_X = array(train_x)/height

    model = Model()

    model.add(Input(2))
    model.add(Dense(120, 'sigmoid'))
    model.add(Dense(90, 'sigmoid'))
    model.add(Dense(1, 'sigmoid'))

    model.compile()

    model.fit(train_X,
              train_Y,
              epochs=1000 + len(train_x)*200,
              view_stat=False)
    upload_img()

def upload_img():

    photoshop = Image.new("RGB", (height, width))
    pixels = photoshop.load()

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

    for i,(x,y) in enumerate(x_list):
        pixels[int(y),int(x)] = ((1-y_list[i])*255,
                                 (1-y_list[i])*255,
                                 (1-y_list[i])*255)

    img_photo = ImageTk.PhotoImage(photoshop)

    label["image"] = img_photo
    label.image = img_photo

if __name__ == '__main__':

    win = tkinter.Tk()

    win.geometry(f'{width}x{height}')

    win.bind("<Button-1>", callback_1)
    win.bind("<Button-2>", callback_2)
    win.bind("<Button-3>", callback_3)

    label = tkinter.Label(win)
    label.pack()

    win.mainloop()