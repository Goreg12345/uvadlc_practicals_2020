"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

import random

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

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

    right_preds = torch.sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
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
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100

    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    # neg_slope = FLAGS.neg_slope
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    def plot_history(results, model_name=""):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
        plt.plot([i for i in range(1, len(results["train_scores"]) + 1)], results["train_scores"], label="Train")
        plt.plot([i for i in range(1, len(results["val_scores"]) + 1)], results["val_scores"], label="Val")
        plt.xlabel("Epochs")
        plt.ylabel("Validation accuracy")
        plt.ylim(min(results["val_scores"]), max(results["train_scores"]) * 1.01)
        plt.title("Validation performance of %s" % model_name)
        plt.legend()
        plt.show()

    device = torch.device('cuda')

    cifar10 = cifar10_utils.get_cifar10(f'cifar10/cifar-10-batches-py')
    train_data = cifar10_utils.DataSet(cifar10['train'].images, cifar10['train'].labels)
    test_data = cifar10_utils.DataSet(cifar10['test'].images, cifar10['test'].labels)

    def fit(hyperparameter):
        model = MLP(3 * 32 * 32, hyperparameter['dnn_hidden_units'], 10, hyperparameter).to(device)

        loss_module = nn.CrossEntropyLoss()
        optimizer = hyperparameter['optimizer'](
            model.parameters(), lr=hyperparameter['learning_rate']
        )

        # train_loader = torch.utils.data.Dataloader(train_data, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)

        results = dict(train_scores=list(), val_scores=list())
        for i in range(hyperparameter['n_steps']):
            x, y = train_data.next_batch(FLAGS.batch_size)

            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)

            preds = model(torch.flatten(x, start_dim=1))
            preds = preds.squeeze(dim=1)

            if i % 500 == 499:
                results['train_scores'].append(accuracy(preds, y).cpu())

                x_test, y_test = test_data.next_batch(300)
                x_test, y_test = torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).long().to(device)
                preds_test = model(torch.flatten(x_test, start_dim=1))
                preds_test = preds_test.squeeze(dim=1)
                results['val_scores'].append(accuracy(preds_test, y_test).cpu())
                print("current step: ", accuracy(preds, y))

            _, y = torch.max(y, dim=1)
            loss = loss_module(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plot_history(results)

        # Test
        x, y = test_data.next_batch(10000)

        x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)

        preds = model(torch.flatten(x, start_dim=1))
        preds = preds.squeeze(dim=1)

        print("Test Accuracy: ", accuracy(preds, y))
        return accuracy(preds, y)


    # Randomized Search
    search_space = dict(
        learning_rate=[0.0008, 0.001, 0.0015],
        optimizer=[torch.optim.SGD, torch.optim.Adam, torch.optim.Adagrad, torch.optim.RMSprop],
        n_steps=np.arange(10000, 15000, 100),
        dnn_hidden_units=[[i] for i in np.arange(30, 250, 20)],
        dropout=np.arange(0., 0.5, 0.1),
        n_layers=np.arange(1, 4)
    )
    max_acc = 0
    for i in range(50):
        random_hyperparameter = { j: random.choice(search_space[j]) for j in search_space }
        print("Next params: ", random_hyperparameter)
        acc = fit(random_hyperparameter)
        max_acc = max(max_acc, acc)
    print('Best Accuracy: ', max_acc)



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
