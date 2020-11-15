"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        super(ConvNet, self).__init__()

        class PreAct(nn.Module):
            def __init__(self, n_channels):
                super(PreAct, self).__init__()
                self.batch_norm = nn.BatchNorm2d(n_channels)
                self.relu = nn.ReLU()
                self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)

            def forward(self, x):
                res = self.batch_norm(x)
                res = self.relu(res)
                res = self.conv(res)
                return x + res

        self.conv0 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.PreAct1 = PreAct(64)

        self.conv1 = nn.Conv2d(64, 128, 1)
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)
        self.PreAct2_a = PreAct(128)
        self.PreAct2_b = PreAct(128)

        self.conv2 = nn.Conv2d(128, 256, 1)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)
        self.PreAct3_a = PreAct(256)
        self.PreAct3_b = PreAct(256)

        self.conv3 = nn.Conv2d(256, 512, 1)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)
        self.PreAct4_a = PreAct(512)
        self.PreAct4_b = PreAct(512)

        self.maxpool4 = nn.MaxPool2d(3, 2, 1)
        self.PreAct5_a = PreAct(512)
        self.PreAct5_b = PreAct(512)
        self.maxpool5 = nn.MaxPool2d(3, 2, 1)

        self.linear = nn.Linear(512, n_classes)

        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        x = self.conv0(x)
        x = self.PreAct1(x)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.PreAct2_a(x)
        x = self.PreAct2_b(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.PreAct3_a(x)
        x = self.PreAct3_b(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.PreAct4_a(x)
        x = self.PreAct4_b(x)

        x = self.maxpool4(x)
        x = self.PreAct5_a(x)
        x = self.PreAct5_b(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, start_dim=1)
        out = self.linear(x)

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
