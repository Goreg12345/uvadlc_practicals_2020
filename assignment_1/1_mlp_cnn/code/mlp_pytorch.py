"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes, hyperparameter=dict(n_layers=1, dropout=0)):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
    
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        super(MLP, self).__init__()
        layers = list()
        layers.append(nn.Dropout(0.5))
        for n_hidden_neurons in n_hidden * hyperparameter['n_layers']:
            layers.append(nn.Linear(n_inputs, n_hidden_neurons))
            layers.append(nn.ELU())
            n_inputs = n_hidden_neurons
        layers.append(nn.Dropout(hyperparameter['dropout']))
        layers.append(nn.Linear(n_inputs, n_classes))
        # layers.append(nn.Softmax())

        self.model = nn.Sequential(*layers)

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

        out = self.model(x)

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
