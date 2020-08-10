#-------------------------------#
#       Author: grafstor        
#       Date: 12.06.20          
#-------------------------------#

import time
from tkinter import Tk, Label

from geras import *
from PIL import Image, ImageTk

class Paint:
    def __init__(self, root):
        self.height = 100
        self.width = 100

        self.x_list = []
        self.y_list = []

        self.model = 0


        root.geometry(f'{self.width}x{self.height}')

        root.bind("<Button-1>", self.button_1)
        root.bind("<Button-2>", self.button_2)
        root.bind("<Button-3>", self.button_3)

        self.label = Label(root)
        self.label.pack()

    def button_1(self, event):
        self.x_list.append([event.y, event.x])
        self.y_list.append(1)

        self.new_model(self.x_list, self.y_list)

    def button_2(self, event):
        train_X = np.array(self.x_list)/self.height
        train_Y = np.array(self.y_list)
        train_Y = train_Y.reshape((train_Y.shape[0], 1))

        epochs = 1000 + len(self.x_list)*190
        
        for eph in range(epochs):
            self.model.train(train_X, train_Y)

        self.upload_img()

    def button_3(self, event):
        self.x_list.append([event.y, event.x])
        self.y_list.append(0)

        self.new_model(self.x_list, self.y_list)

    def new_model(self, train_x, train_Y):
        global model

        train_X = np.array(train_x)/self.height
        train_Y = np.array(train_Y)
        train_Y = train_Y.reshape((train_Y.shape[0], 1))

        self.model = Model(

            Dense(120),
            Sigmoid(),

            Dense(90),
            Sigmoid(),

            Dense(1),
            Sigmoid(),

        )(Adam(0.01))

        epochs = 2000 + len(train_x)*200

        for eph in range(epochs):
            self.model.train(train_X, train_Y)

        self.upload_img()

    def upload_img(self):

        photoshop = Image.new("RGB", (self.height, self.width))
        pixels = photoshop.load()

        test = []
        for x in range(self.width):
            for y in range(self.height):
                test.append([y,x])

        result = self.model.predict(np.array(test)/self.height)
        result = np.array(result)
        result = result.reshape(self.width, self.height,1)

        for x in range(self.width):
            for y in range(self.height):
                dot = result[y,x]
                pixels[y, x] = (int(dot*255),
                                int(dot*255),
                                int(dot*255))

        for i,(x,y) in enumerate(self.x_list):
            pixels[int(y),int(x)] = ((1-self.y_list[i])*255,
                                     (1-self.y_list[i])*255,
                                     (1-self.y_list[i])*255)

        img_photo = ImageTk.PhotoImage(photoshop)

        self.label["image"] = img_photo
        self.label.image = img_photo

def main():
    root = Tk()
    Paint(root)
    root.mainloop()

if __name__ == '__main__':
    main()