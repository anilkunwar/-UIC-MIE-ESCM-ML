import numpy as np

import tensorflow as


class Generator(tf.keras.Model):

    def __init__(self,z_size,input_size_G,input_channels_G,output_channels_G):

        super(Generator, self).__init__()

        self.z_size=z_size

        self.input_size_G=input_size_G

        self.input_channels_G=input_channels_G

        self.output_channels_G=output_channels_G

        self.fc=tf.keras.layers.Dense(self.input_channels_G*self.input_size_G[0]*self.input_size_G[0])

        self.conv_transp_1=tf.keras.layers.Conv2DTranspose(
                                              self.input_channels_G//2,
                                              kernel_size=4,
                                              strides=2,
                                              padding='same',
                                              use_bias=False)

        self.batch_norm_1= tf.keras.layers.BatchNormalization()

        self.conv_transp_2=tf.keras.layers.Conv2DTranspose(
                                              self.input_channels_G//4,
                                              kernel_size=4,
                                              strides=2,
                                              padding='same',
                                              use_bias=False)


        self.batch_norm_2= tf.keras.layers.BatchNormalization()

        self.conv_transp_3=tf.keras.layers.Conv2DTranspose(
                                              self.input_channels_G//8,
                                              kernel_size=4,
                                              strides=2,
                                              padding='same',
                                              use_bias=False)


        self.batch_norm_3= tf.keras.layers.BatchNormalization()

        self.conv_transp_4=tf.keras.layers.Conv2DTranspose(
                                              self.input_channels_G//16,
                                              kernel_size=4,
                                              strides=2,
                                              padding='same',
                                              use_bias=False)


        self.batch_norm_4= tf.keras.layers.BatchNormalization()

        self.conv_transp_5=tf.keras.layers.Conv2DTranspose(
                                              self.output_channels_G,
                                              kernel_size=4,
                                              strides=2,
                                              padding='same',
                                              use_bias=False)


    def call(self,x):

        x=self.fc(x)

        x=tf.reshape(x,(-1,self.input_size_G[0],self.input_size_G[0],self.input_channels_G))

        x=tf.keras.activations.relu(self.batch_norm_1(self.conv_transp_1(x)))

        x=tf.keras.activations.relu(self.batch_norm_2(self.conv_transp_2(x)))

        x=tf.keras.activations.relu(self.batch_norm_3(self.conv_transp_3(x)))

        x=tf.keras.activations.relu(self.batch_norm_4(self.conv_transp_4(x)))

        x=tf.keras.activations.tanh(self.conv_transp_5(x))

        return x

class Discriminator(tf.keras.Model):

    def __init__(self,input_size_G,input_channels_D,output_channels_D):

        super (Discriminator,self).__init__()

        self.input_size_G=input_size_G

        self.input_channels_D=input_channels_D

        self.output_channels_D=output_channels_D

        self.conv_1=tf.keras.layers.Conv2D(self.input_channels_D,
                                          kernel_size=4,
                                          strides=2,
                                          padding='same',
                                          use_bias='False')

        self.conv_2=tf.keras.layers.Conv2D(self.input_channels_D*2,
                                          kernel_size=4,
                                          strides=2,
                                          padding='same',
                                          use_bias='False')

        self.batch_norm_2=tf.keras.layers.BatchNormalization()

        self.conv_3=tf.keras.layers.Conv2D(self.input_channels_D*4,
                                          kernel_size=4,
                                          strides=2,
                                          padding='same',
                                          use_bias='False')

        self.batch_norm_3=tf.keras.layers.BatchNormalization()

        self.conv_4=tf.keras.layers.Conv2D(self.input_channels_D*8,
                                          kernel_size=4,
                                          strides=2,
                                          padding='same',
                                          use_bias='False')

        self.batch_norm_4=tf.keras.layers.BatchNormalization()

        self.conv_5=tf.keras.layers.Conv2D(self.input_channels_D*16,
                                          kernel_size=4,
                                          strides=2,
                                          padding='same',
                                          use_bias='False')

        self.batch_norm_5=tf.keras.layers.BatchNormalization()

        self.fc=tf.keras.layers.Dense(output_channels_D)



    def call(self,x):

        x=tf.keras.layers.LeakyReLU(alpha=0.2)(self.conv_1(x))

        x=tf.keras.layers.LeakyReLU(alpha=0.2)(self.batch_norm_2(self.conv_2(x)))

        x=tf.keras.layers.LeakyReLU(alpha=0.2)(self.batch_norm_3(self.conv_3(x)))

        x=tf.keras.layers.LeakyReLU(alpha=0.2)(self.batch_norm_4(self.conv_4(x)))

        x=tf.keras.layers.LeakyReLU(alpha=0.2)(self.batch_norm_5(self.conv_5(x)))

        x=tf.reshape(x,(-1,self.input_channels_D*input_size_G[0]*input_size_G[0]*16))

        x=tf.keras.activations.sigmoid(self.fc(x))

        return (x)


batch_size=64
image_size=64

# Defining the Generator
z_size=100

num_conv_layers=5

input_size_G=(image_size//2**num_conv_layers,)*2

input_channels_G=512

output_channels_G=3

# Definiing the Discriminator
input_channels_D=64

output_channels_D=1

G=Generator(z_size=z_size,
            input_size_G=input_size_G,
            input_channels_G=input_channels_G,
            output_channels_G=output_channels_G)

D=Discriminator(input_size_G=input_size_G,
                input_channels_D=input_channels_D,
               output_channels_D=output_channels_D
               )
