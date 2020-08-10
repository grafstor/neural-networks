#-------------------------------#
#       Author: grafstor        
#       Date: 22.07.20          
#-------------------------------#

from geras import *

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class GAN():
    def __init__(self, latent_dim=100):
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
            BatchNormalization(0.8),

            Dense(512),
            LeakyReLU(0.2),
            BatchNormalization(0.8),

            Dense(1024),
            LeakyReLU(0.2),
            BatchNormalization(0.8),

            Dense(np.prod(self.img_shape)),
            TanH(),

            Reshape(self.img_shape),

        )(Adam(0.0003, 0.5))

        return generator

    def build_discriminator(self):

        discriminator = Model(

            Flatten(),

            Dense(512),
            LeakyReLU(0.2),

            Dense(256),
            LeakyReLU(0.2),

            Dense(1),
            Sigmoid(),

        )(Adam(0.002, 0.5))

        return discriminator

    def train(self, x_train, epochs=10000, interval=200):

        self.build(x_train.shape[1:])

        batch_size = 256

        zeros = np.zeros((batch_size, 1))
        ones = np.ones((batch_size, 1)) - 0.1


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
                self.show_samples(epoch, x_train.shape[1])

    def show_samples(self, epoch, size):
        r, c = 3, 3

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = gen_imgs.reshape((-1, size, size, 1))

        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig("gen_pics/%d.png" % epoch)
        plt.close()


def load_data(path):
    train = pd.read_csv(path+"/train.csv")
    train_x = train.drop(labels=["label"], axis=1)

    train_x = np.array(train_x)

    train_x = train_x.reshape((-1, 28, 28, 1))
    train_x = train_x / 127.5 - 1.

    return train_x

def main():

    x_train = load_data('test train data/mnist')

    gan = GAN()
    gan.train(x_train)

if __name__ == '__main__':
    main()
