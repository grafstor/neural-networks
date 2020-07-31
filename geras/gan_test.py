#-------------------------------#
#       Author: grafstor        
#       Date: 22.07.20          
#-------------------------------#

from geras import *

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class GAN():
    def __init__(self, latent_dim):
        self.img_shape = (28, 28, 1)
        self.latent_dim = latent_dim

    def build(self, img_shape):
        self.img_shape = img_shape

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.combined = Model(*self.generator.layers, *self.discriminator.layers)

    def build_generator(self):

        generator = Model(

            Dense(256),
            LeakyReLU(0.2),
            Dropout(0.5),

            Dense(512),
            LeakyReLU(0.2),
            Dropout(0.5),

            Dense(1024),
            LeakyReLU(0.2),
            Dropout(0.5),

            Dense(np.prod(self.img_shape)),
            TanH(),

            Reshape(self.img_shape),

        )(Adam(0.0012, 0.5))

        return generator

    def build_discriminator(self):

        discriminator = Model(

            Flatten(self.img_shape),

            Dense(512),
            LeakyReLU(0.2),

            Dropout(0.25),

            Dense(256),
            LeakyReLU(0.2),

            Dropout(0.25),

            Dense(1),
            Sigmoid(),

        )(Adam(0.0012, 0.5))

        return discriminator

    def train(self, x_train, epochs=2000, interval=200):

        self.build(x_train.shape[1:])

        batch_size = 128

        zeros = np.zeros((batch_size, 1))
        ones = np.ones((batch_size, 1))


        for epoch in range(epochs):

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_image = self.generator.predict(noise)

            ids = np.random.randint(0, x_train.shape[0], batch_size)
            true_image = x_train[ids]

            for layer in self.discriminator.layers:
                layer.trainable = True
            


            gen_result = self.discriminator.train(gen_image, zeros)
            true_result = self.discriminator.train(true_image, ones)


            d_loss_fake = np.mean(self.discriminator.loss.loss(zeros, gen_result))
            d_loss_real = np.mean(self.discriminator.loss.loss(ones, true_result))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            for layer in self.discriminator.layers:
                layer.trainable = False



            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            comb_result = self.combined.train(noise, ones)


            g_loss = np.mean(self.combined.loss.loss(ones, comb_result))

            print('epoch', epoch, ' discriminator:', d_loss, ' generator:', g_loss)

            if epoch%interval == 0:
                image_show(epoch, gen_image[0], x_train.shape[1])


def load_data(path):
    train = pd.read_csv(path+"/train.csv")
    train_x = train.drop(labels=["label"], axis=1)

    train_x = np.array(train_x)

    train_x = train_x.reshape((-1, 28, 28, 1))
    train_x = train_x / 127.5 - 1.

    return train_x

def image_show(epc, img, size):
    plt.imshow(img.reshape((size,size)), cmap='gray')
    plt.savefig(f'gen_pics/{epc}.png')

def main():

    x_train = load_data('test train data/mnist')

    gan = GAN(100)
    gan.train(x_train)

if __name__ == '__main__':
    main()
