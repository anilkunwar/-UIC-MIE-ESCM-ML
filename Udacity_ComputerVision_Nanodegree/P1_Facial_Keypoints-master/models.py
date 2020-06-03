## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self,image_size,input_channels,output_channels,n_conv_layers):
        
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.image_size=image_size[0]
        
        self.input_channels=input_channels
        
        self.output_channels=output_channels
        
        self.n_conv_layers=n_conv_layers
        
        self.kernel_size=3
        
        self.conv_1 = nn.Conv2d(1, self.input_channels,self.kernel_size)
        
        self.batch_norm_1=nn.BatchNorm2d(self.input_channels)
        
        self.conv_2 = nn.Conv2d(self.input_channels,self.input_channels*2,self.kernel_size)
        
        self.batch_norm_2=nn.BatchNorm2d(self.input_channels*2)
        
        self.conv_3 = nn.Conv2d(self.input_channels*2,self.input_channels*4,self.kernel_size)
        
        self.batch_norm_3=nn.BatchNorm2d(self.input_channels*4)
        
        self.conv_4 = nn.Conv2d(self.input_channels*4,self.input_channels*8,self.kernel_size)
        
        self.batch_norm_4=nn.BatchNorm2d(self.input_channels*8)
        
        self.conv_5 = nn.Conv2d(self.input_channels*8,self.input_channels*16,self.kernel_size)
        
        self.batch_norm_5=nn.BatchNorm2d(self.input_channels*16)
        
        self.conv_6 = nn.Conv2d(self.input_channels*16,self.input_channels*32,self.kernel_size)
        
        self.batch_norm_6=nn.BatchNorm2d(self.input_channels*32)
        
        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.dropout=nn.Dropout(0.5)
        
        self.fc=nn.Linear((self.image_size//(2**self.n_conv_layers)-2)**2*self.input_channels*8,self.output_channels)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
           
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #x=self.max_pool(F.relu(self.batch_norm_1(self.dropout(self.conv_1(x)))))
        x=self.max_pool(F.relu(self.batch_norm_1(self.conv_1(x))))
        
        #x=self.max_pool(F.relu(self.batch_norm_2(self.dropout(self.conv_2(x)))))
        x=self.max_pool(F.relu(self.batch_norm_2(self.conv_2(x))))
        
        #x=self.max_pool(F.relu(self.batch_norm_3(self.dropout(self.conv_3(x)))))
        x=self.max_pool(F.relu(self.batch_norm_3(self.conv_3(x))))
        
        #x=self.max_pool(F.relu(self.batch_norm_4(self.dropout(self.conv_4(x)))))
        x=self.max_pool(F.relu(self.batch_norm_4(self.conv_4(x))))
        
        #x=self.max_pool(F.relu(self.batch_norm_5(self.dropout(self.conv_5(x)))))
        #x=self.max_pool(F.relu(self.batch_norm_5(self.conv_5(x))))
        
        #x=self.max_pool(F.relu(self.batch_norm_6(self.dropout(self.conv_6(x)))))
        #x=self.max_pool(F.relu(self.batch_norm_6(self.conv_6(x))))
        
        x=x.view(-1,(self.image_size//(2**self.n_conv_layers)-2)**2*self.input_channels*8)
       
        x = self.fc(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
