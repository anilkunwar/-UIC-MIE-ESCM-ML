# Deep Convolutional GANs

# Goal: generation of fake images looking at real images


# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
#batchSize = 64 # We set the size of the batch.
batchSize=2
#imageSize = 64 # We set the size of the generated images (64x64).
imageSize=64

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
# CIFAR dataset: well known dataset of images
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    # the function looks all the layers in the network and when it finds
    # modules which contain the names 'Conv' and 'BatchNorm', it initializes
    # the weights and the bias with the values reported below.
    # the modules ConvTranspose2d and Conv2d have both the name 'Conv'
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # the module BatchNorm2d has the name 'BatchNorm'
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator

# creating pytorch class
class G(nn.Module): # We introduce a class to define the generator.

    # self: future object that will be created from the class G. Everytime we use self, we refer to the object of the class.
    # we attach a variable to self to specify that the variable belongs to the object
    def __init__(self): # We introduce the __init__() function that will define the architecture of the generator.

        # super(G,self).__init__() >>> activate the nn.Module for the class and the object'
        # the super function takes in input the class (G) and the object of the class (self)
        super(G, self).__init__() # We inherit from the nn.Module tools.

        # make a meta-module (module which contains a sequence of several modules, \
        # where each module is a layer)
        # self.main >>> meta-module >>> object of the nn.Sequential class
        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).

            # the architecture implemented here is an architecture which has been proven
            # to work well for GANs. The input parameters like the input vector size,
            # number of features maps come from experimentation.

            # inverse convolution: convolutional layer which takes in input a 1d array (vector)
            # and it returns an image. The input to this inverse cnn is a vector of random noise values
            # with a length of 100

            # nn.ConvTranspose2d: pytorch module for inverse convolution'
            # 100: size of the input vector
            # 512: number of output features maps
            # 4: kernel size
            # 1: stride
            # 0: padding
            # bias=False: we do not include the bias
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # We start with an inversed convolution.

            # batch normalization on the 512 features maps
            nn.BatchNorm2d(512), # We normalize all the features along the dimension of the batch.

            # relu rectifief activation function to break the linearities
            nn.ReLU(True), # We apply a ReLU rectification to break the linearity.

            # 512: input to the new inverse convolution = output of the last inverse convolution
            # since the previous inverse convolution has 512 features maps as output, the input dimension
            # of the new inverse convolution must be 512
            # 256: number of features maps as output of the new convolution
            # 4: kernel size
            # 2: stride
            # 1: padding
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # We add another inversed convolution.

            # batch norm the 256 features maps.
            nn.BatchNorm2d(256), # We normalize again.

            nn.ReLU(True), # We apply another ReLU.

            # another transpose convolution
            # 256: input to the new inverse convolution = output of the last inverse convolution
            # since the previous inverse convolution has 256 features maps as output, the input dimension
            # of the new inverse convolution must be 256
            # 128: number of features maps as output of the new convolution
            # 4: kernel size
            # 2: stride
            # 1: padding
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # We add another inversed convolution.

            # batch norm the 128 features maps.
            nn.BatchNorm2d(128), # We normalize again.

            nn.ReLU(True), # We apply another ReLU.

            # another transposed convolution + batch-norm + relu
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # We add another inversed convolution.

            nn.BatchNorm2d(64), # We normalize again.

            nn.ReLU(True), # We apply another ReLU.

            # final inverse convolution
            # 64: number of features maps from the previous inverse convolution
            # 3: number of output features maps of this last inverse convolution
            # since the output of the generator is the image which is given in input
            # to the discriminator, such image must have 3 channels (as all the images),
            # se output channel of the last inverse convolution must be 3
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), # We add another inversed convolution.

            # tanh to generate images with pixel values between -1 and +1
            # the output image of the generator will be the input image of the discriminator
            nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
        )

    # forward function to process the input through the layers
    # self: we need to give in input self (the object) in order to be able to use self.main
    # input: vector of random values with size 100
    def forward(self, input): # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output containing the generated images.

        # applying the neural network defined as the attribute main of the object to the input
        output = self.main(input) # We forward propagate the signal through the whole neural network of the generator defined by self.main.

        return output # We return the output containing the generated images.

# Creating the generator
# netG: object of the class G
netG = G() # We create the generator object.
# applying the initial weights to the neural network with the .apply function
# the weights will be updated in the learning process
netG.apply(weights_init) # We initialize all the weights of its neural network.

# Defining the discriminator

class D(nn.Module): # We introduce a class to define the discriminator.

    def __init__(self): # We introduce the __init__() function that will define the architecture of the discriminator.
        super(D, self).__init__() # We inherit from the nn.Module tools.

        # self.main: object of the Sequential class

        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).

            # first layer of the discriminator: convolotutional layer
            # the D takes in input the image generated by the G
            # 3:number of channels of the input image
            # the value of the input channels of the first layer of the discriminator
            # must be equal to the value of the output channels of the last layer of the generator

            # 64: output channels
            # 4: kernel size
            # 2: stride
            # 1: padding
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), # We start with a convolution.

            # LeakyRelu: x if x>=0  -a*x if x<0 with a<1 >>> negative values for negative x
            # we use LeakyReLU instead of Relu because doing some experimentation and research
            # it has been found that LeakyRelU works better
            # 0.2: coefficient a of the LeakyReLU (slope of the line for x<0)
            nn.LeakyReLU(0.2, inplace = True), # We apply a LeakyReLU.

            # series of conv2d + batch-norm + leaky relu
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # We add another convolution.

            nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.

            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.

            nn.Conv2d(128, 256, 4, 2, 1, bias = False), # We add another convolution.

            nn.BatchNorm2d(256), # We normalize again.

            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.

            nn.Conv2d(256, 512, 4, 2, 1, bias = False), # We add another convolution.

            nn.BatchNorm2d(512), # We normalize again.

            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.

            # 1: output channel of the last convolutiona layer
            # the output of the last convolution of the discriminator
            # is a single feature map
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # We add another convolution.

            # we use Sigmoid activatio function to obtain probabilities between 0 and 1
            # 0: rejection of the image
            # 1: acceptance of the image
            # if a value is <0.5 we turn it to 0 and we reject the image
            # if a value is >=0.5 we turn it to 1 and we accept the image
            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
        )

        # propagatin the input through the neural network
        # self: give in input the object of the class, so we can use the attribute self.main
        # input: image generated by the discriminator
    def forward(self, input): # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1.
        output = self.main(input) # We forward propagate the signal through the whole neural network of the discriminator defined by self.main.

        # output.view(-1)
        # since in the graph the last layer is a convolutional layer,
        # the so far output is a 2d feature map
        # since we want to return a scalar (probability of acceptance),
        # we must flattend the output of the convolution in a way that the final output
        # is a 1d array with the followign shape:

        # (batch_size,)

        # so each element of the output array is the probability of acceptance
        # for each image in the batch entering in the neural network
        return output.view(-1) # We return the output which will be a value between 0 and 1.

# Creating the discriminator
netD = D() # We create the discriminator object.
netD.apply(weights_init) # We initialize all the weights of its neural network.

# Training the DCGANs

# BCELoss: Bynary Cross Entropy loss for estimations with values between 0 and 1
criterion = nn.BCELoss() # We create a criterion object that will measure the error between the prediction and the target.

# optimizers for the D and the G (optimizer: methods used to update the weights)
# netD.parameters(),netG.parameters: parameters of the neural network
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We create the optimizer object of the discriminator.
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We create the optimizer object of the generator.

for epoch in range(25): # We iterate over 25 epochs.



    # i: index of the loop
    # data: minibatch of images
    # enumerate: function to get the minibatches from dataloader
    # dataloader: object from which we get the minibatches
    # 0: the index of the loop starts from 0
    for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset.

        # 1st Step: Updating the weights of the neural network of the discriminator

        netD.zero_grad() # We initialize to 0 the gradients of the discriminator with respect to the weights.

        # the Discriminator must be trained both on the real images and
        # the fake images created by the generator

        # Training the discriminator with a REAL IMAGE of the dataset

        # data: the first element of data are the images, the second element of data are the corresponding labels
        # real, _ = data: get the first element of data (the images)
        real, _ = data # We get a real image of the dataset which will be used to train the discriminator.

        # define the input minibatch and the target as pytorch Variable
        # >>> object with a gradient function grad_fn
        input = Variable(real) # We wrap it in a variable.

        # since we are training the discriminator with the REAL IMAGES
        # and we want the model to learn that those images are real,
        # the corresponding label is 1
        # 1: acceptance
        # 0: rejection
        # torch.ones(input.size()[0]): vector of 1s with a size equal to the number of images in the minibatch
        # >>> for each real image the label is 1
        # Variable: we store the target into a torch Variable which is characterized
        # by a gradient function grad_fn, since we will calculate a gradient which involves
        # the target
        target = Variable(torch.ones(input.size()[0])) # We get the target.

        # get the prediction from the neural network of the discriminator
        # the output is a vector of 0 and 1 with a size equal to the number of images
        # in the minibatch. For each REAL IMAGE, the discriminator is trained
        # to predict 1 (acceptance)
        output = netD(input) # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).

        # calculating the error using criterion
        # _real: discriminator trained on REAL images
        errD_real = criterion(output, target) # We compute the loss between the predictions (output) and the target (equal to 1).

        # Training the discriminator with a FAKE generated by the generator
        # the input of the Discriminator is a minibatch of fake images created by the Generator
        # so the input of the Generator must be a minibatch containing
        # a number equal to the batch size of random vectors with size 100

        # the input of the Generator must be a vector which enters
        # in the inverse convolution of the discriminator

        # input.size()[0]: batch size
        # 100: size of each random vector
        # 1,1: fake dimensions to create an array of 100 elements
        # in this way each element is a matrix of 1x1
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # We make a random input vector (noise) of the generator.

        # fake image created by the Generator neural network giving the noise as input
        fake = netG(noise) # We forward propagate this random input vector into the neural network of the generator to get some fake generated images.

        # since we are training the discriminator with the FAKE IMAGES
        # and we want the model to learn that those images are fake,
        # the corresponding label is 0
        # 1: acceptance
        # 0: rejection
        # torch.zeros(input.size()[0]): vector of 1s with a size equal to the number of images in the minibatch
        # >>> for each real image the label is 0
        # Variable: we store the target into a torch Variable which is characterized
        # by a gradient function grad_fn, since we will calculate a gradient which involves
        # the target
        target = Variable(torch.zeros(input.size()[0])) # We get the target.

        # fake.detach())since we do not update the weights of the Generator in this step (we are training the Discriminator)
        # we are not interested in the gradients associated with the output of the Generator
        # so we detach the gradients from fake to save memory
        output = netD(fake.detach()) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).

        # calculating the error on the prediction of the fake image
        errD_fake = criterion(output, target) # We compute the loss between the prediction (output) and the target (equal to 0).

        # Backpropagating the total error

        # the total error takes into account the error obtained on the prediction
        # for the real image and for the fake image
        # remind:
        # errD_real: error calculated between the prediction for the REAL image and the target (1 >>> acceptance of a REAL image)
        # errD_fake: error calculated between the prediction for the FAKE image and the target (0 >>> rejection of a FAKE image)
        errD = errD_real + errD_fake # We compute the total error of the discriminator.

        # backpropagation
        errD.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.

        # updating the weights of the Discriminator
        optimizerD.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

        # 2nd Step: Updating the weights of the neural network of the generator

        netG.zero_grad() # We initialize to 0 the gradients of the generator with respect to the weights.

        # the target now is 1 since we want to train the Generator to produce
        # images which look like to REAL images
        target = Variable(torch.ones(input.size()[0])) # We get the target.

        # the training of the Generator is performed in such a way that the prediction
        # of the Discriminator when taking as input the FAKE image generated by the Generator
        # is very close to a REAL image
        # this time we do not detach the gradients from fake since we want to
        # update the weights of the Generator respect to these gradients
        output = netD(fake) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).

        # the error is calculated between the prediction of the Discriminator
        # when taking in input the fake from the generator and the corresponding vector of 1s
        # (1 >>> REAL IMAGES)
        errG = criterion(output, target) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).

        # backpropagation
        errG.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.

        # updating the weights
        optimizerG.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.

        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

        #print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).

        # save REAL and FAKE images every 100 steps
        # using vutils, a pytorch library for computer vision

        #if i % 100 == 0: # Every 100 steps:
        if i % 1000 == 0: # Every 100 steps:

            # save the REAL image
            #vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            # get the FAKE image
            fake = netG(noise) # We get our fake generated images.
            # save the FAKE image
            #vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%0d_%03d.png' % ("./results",i, epoch), normalize = True) # We also save the fake generated images of the minibatch.

        print('step {}/{} epoch {}'.format(i,len(dataset.data)//batchSize,epoch))
