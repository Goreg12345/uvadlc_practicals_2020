"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.params = dict(
            weight=np.random.normal(loc=0, scale=0.0001, size=(out_features, in_features)),
            bias=np.zeros((1, out_features))
        )
        self.result_forward = None
        self.x = None
        self.grads = None
        
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = x
        self.result_forward = x @ self.params['weight'].T + self.params['bias']
        out = self.result_forward
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dL_dY = dout
        dL_dW = dL_dY.T @ self.x
        # dL_db = dL_dY @ np.ones(dL_dY.shape[1])
        dL_db = np.ones((1, self.x.shape[0])) @ dL_dY
        dL_dX = dL_dY @ self.params['weight']

        self.grads = dict(
            weight=dL_dW,
            bias=dL_db
        )
        dx = dL_dX
        
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        def exp_normalize(x):
            b = x.max(axis=1, keepdims=True)
            y = np.exp(x - b)
            return y / np.sum(y, axis=1, keepdims=True)

        # exp = np.exp(x)
        # divisor = np.sum(exp, axis=1)
        # out = exp / divisor

        out = exp_normalize(x)
        self.out = out

        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = list()
        for i in range(dout.shape[0]):
            dy_dx = np.multiply(self.out[i], np.identity(self.out.shape[1]) - self.out[i])
            dx.append(dout[i] @ dy_dx)
        dx = np.asarray(dx)
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        s = x.shape[0]
        out = (-1 / s) * np.sum(np.multiply(y, np.log(x)))
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        s = x.shape[0]
        dx = (-1 / s) * np.multiply(y, 1 / x)
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        out = np.where(x < 0, np.exp(x) - 1, x)
        self.x = x
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = np.multiply(dout, np.where(self.x < 0, np.exp(self.x), 1))

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
