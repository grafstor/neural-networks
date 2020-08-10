#-------------------------------#
#       Author: grafstor        
#       Date: 09.08.20          
#-------------------------------#

import time
import tkinter as tk

from geras import *
from PIL import Image, ImageTk, ImageDraw
import pickle

class Drawer:
    def __init__(self, root):
        self.height = 280
        self.width = 280

        self.image = None
        self.draw = None

        self.past_x = 0
        self.past_y = 0

        with open('mnist_model.pkl', 'rb') as model:
            self.model = pickle.load(model)

        self.root = root
        self.root.geometry(f'{self.width}x{self.height}+100+100')
        self.root.bind('<B1-Motion>', self.motion)
        self.root.bind('<Button-3>', self.new)

        self.label = tk.Label(root, width=self.width, height=self.height)
        self.label.pack(side=tk.LEFT)

        self.new_picture()

        print('Right Mouse Button - draw')
        print('Left Mouse Button - clean')

    def new(self, event):
        self.new_picture()
        self.update_picture(self.image)

        self.past_x = 0
        self.past_y = 0

    def motion(self, event):
        x, y = event.x, event.y

        if self.past_x:
            self.draw_line(self.past_x, self.past_y, x, y)

        self.past_x = x
        self.past_y = y

        self.update_picture(self.image)

        if self.past_x:
            self.predict(self.image)

    def predict(self, image):
        nn_pic = copy.copy(image)
        nn_pic.thumbnail((28,28))

        nn_pic = np.array(nn_pic)[:,:,:1]
        nn_pic = nn_pic.reshape(1, 784)
        test_result = self.model.predict(nn_pic)[0]

        print('You draw -', np.argmax(test_result, axis=0), end='\r')

    def draw_line(self,xx, yy, x, y):
        hr = self.width//13
        r = hr//2
        self.draw.ellipse([(x-r,y-r),(x+r,y+r)], fill=(255,255,255,255))
        self.draw.line((xx,yy,x,y), fill=(255, 255, 255), width=hr)

    def new_picture(self):
        self.image = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    def update_picture(self, picture):
        picture = ImageTk.PhotoImage(picture)

        self.label["image"] = picture
        self.label.image = picture

def main():
    root = tk.Tk()
    Drawer(root)
    root.mainloop()

if __name__ == '__main__':
    main()
