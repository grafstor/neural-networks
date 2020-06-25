#----------------------------#
# Author: grafstor
# Date: 24.06.20
#----------------------------#


__version__ = "1.0"

from geras import Input, Dense, Model
import tkinter as tk

class Viewer(tk.Frame):

    def __init__(self):

        self.width = 1000
        self.height = 800

        self.root = tk.Tk()
        self.root.title("Neural Network")  
        self.root.geometry(f'{self.width+200}x{self.height}+200+100')

        tk.Frame.__init__(self, self.root)

        self.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(self,
                                width=self.width,
                                height=self.height,
                                bg=self.__from_rgb((15,15,15)),
                                bd=0,
                                highlightthickness=0,
                                relief='ridge')

        self.canvas.pack(fill=tk.BOTH,
                         expand=1)

        self.circle_color = self.__from_rgb((113, 0, 173))
        self.font_color = self.__from_rgb((200, 200, 200))

        self.neurons_len = 10
        self.layers_len = 20
        self.neuron_radius = 5
        self.width_max = 6

        self.past_frame = []
        self.past_circle_frame = []

        self.weights = []
        self.layers = []

    def build(self, weights):

        if self.past_frame:
            self.__delete_past_frames()

        else:
            self.__set_neurons_positions(weights)

        self.past_frame = []

        max_weigths = [weight.max() for weight in weights]
        min_weigths = [abs(weight.min()) for weight in weights]

        for i, weight in enumerate(weights):

            from_layer = self.layers[i]
            to_layer = self.layers[i+1]

            for j, coords1 in enumerate(from_layer):
                for k, coords2 in enumerate(to_layer):

                    syn_num = weight[j][k]+min_weigths[i]
                    syn_max = max_weigths[i]+min_weigths[i]

                    delta = syn_num/syn_max

                    width = self.width_max * delta
                    grey = int(235*delta+20)

                    color = self.__from_rgb((grey, 0, grey))

                    self.past_frame.append(self.__line(coords1, coords2, width, color))

        self.__update_circles()

        self.root.update()

    def __set_neurons_positions(self, weights):
        offset_x = (self.width-((len(weights)+1)*self.layers_len))//2-90

        self.layers_len = 700//(len(weights)+1)
        max_h = max([max(i.shape) for i in weights])
        self.neurons_len = 500//max_h


        first_case = weights[0].shape[0]
        offset_y = (self.height-(first_case*self.neurons_len))//2+40

        self.layers = [[] for _ in range(len(weights)+1)]


        for i in range(first_case):
            x = offset_x
            y = offset_y + self.neurons_len*i
            c1 = (x, y)

            self.layers[0].append(c1)

            self.past_circle_frame.append(self.__circle(c1, self.neuron_radius))


        for i in range(len(weights)):
            x = offset_x + self.layers_len*(i+1)

            for j in range(weights[i].shape[1]):

                offset_y = (self.height-(weights[i].shape[1]*self.neurons_len))//2+40

                y = offset_y + self.neurons_len*j

                c1 = (x, y)

                self.layers[i+1].append(c1)

                self.past_circle_frame.append(self.__circle(c1, self.neuron_radius))

    def __update_circles(self):
        for layer in self.layers:
            for c in layer:
                self.past_circle_frame.append(self.__circle(c, self.neuron_radius))

    def __delete_past_frames(self):
        for line in self.past_frame:
            self.canvas.delete(line)
            
        for circle in self.past_circle_frame:
            self.canvas.delete(circle)

    def __line(self, coords1, coords2, width, color):
        return self.canvas.create_line(coords1[0], coords1[1],
                                       coords2[0], coords2[1],
                                       width=width,
                                       fill=color)

    def __circle(self, coords, radius):
        r = radius
        x = coords[0]
        y = coords[1]
        return self.canvas.create_oval(x-r, y-r,
                                       x+r, y+r,
                                       fill=self.circle_color,
                                       outline=self.circle_color)

    def __from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb

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

    viewer = Viewer()

    for i in range(1000):
        model.fit(x, y, epochs=2, view_error=False, view_stat=False)

        weights = [dense.weights for dense in model.layers]
        viewer.build(weights)

    test = [[1,0,1]]
    result = model.predict(test)
    print(result[0])

if __name__ == '__main__':
    main()