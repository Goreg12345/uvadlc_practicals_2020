"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = f'cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    right_preds = np.sum(np.argmax(predictions, dim=1) == np.argmax(targets, dim=1))
    accuracy = right_preds.float() / float(predictions.shape[0])
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    class SGD(object):
        def __init__(self, layers, learning_rate):
            self.layers = layers
            self.learning_rate = learning_rate

        def step(self):
            for layer in self.layers:
                try:
                    layer.grads
                except layer.grads.DoesNotExist:
                    continue
                layer.params['weight'] -= self.learning_rate * layer.grads['weight']
                layer.params['bias'] -= self.learning_rate * layer.grads['bias']

    def eval(model):
        x, y = test_data.next_batch(1000)
        preds = model.forward(np.reshape(x, (x.shape[0], -1)))
        preds = np.flatten(preds)

        loss = loss_module.forward(preds, y)
        print("Test Loss", loss)
        print("Test Accuracy: ", accuracy(preds, y))

    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_data = cifar10_utils.DataSet(cifar10['train'].images, cifar10['train'].labels)
    test_data = cifar10_utils.DataSet(cifar10['test'].images, cifar10['test'].labels)

    model = MLP(3 * 32 * 32, dnn_hidden_units, 10)

    loss_module = CrossEntropyModule()
    optimizer = SGD(model.layers, FLAGS.learning_rate)

    for i in range(FLAGS.max_steps):
        x, y = train_data.next_batch(FLAGS.batch_size)
        preds = model.forward(np.reshape(x, (FLAGS.batch_size, -1)))

        loss = loss_module.forward(preds, y)
        print(loss)
        model.backward(loss_module.backward(preds, y))

        optimizer.step()

        if i % FLAGS.eval_freq == FLAGS.eval_freq - 1:
            eval(model)

    eval(model)

    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
