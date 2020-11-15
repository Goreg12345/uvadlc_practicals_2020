"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torchvision

from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    def plot_history(results, ylabel, title="Validation performance of ", model_name=""):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
        plt.plot([i for i in range(1, len(results["train_scores"]) + 1)], results["train_scores"], label="Train")
        plt.plot([i for i in range(1, len(results["val_scores"]) + 1)], results["val_scores"], label="Val")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.ylim(min(results["val_scores"]), max(results["train_scores"]) * 1.01)
        plt.title(title + model_name)
        plt.legend()
        plt.show()

    def evaluate(model, x_train, y_train, x_test, y_test, loss_module, accuracies, losses):
        x_train, y_train = torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).long().to(device)
        x_test, y_test = torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).long().to(device)

        with torch.no_grad():
            preds = model(x_train)
            preds = preds.squeeze(dim=1)
            accuracies['train_scores'].append(accuracy(preds, y_train).cpu())
            print("current step: ", accuracy(preds, y_train))

            _, y_train = torch.max(y_train, dim=1)
            losses['train_scores'].append(loss_module(preds, y_train).cpu())

            preds = model(x_test)
            preds = preds.squeeze(dim=1)
            accuracies['val_scores'].append(accuracy(preds, y_test).cpu())
            print("current val accuracy: ", accuracy(preds, y_test))

            _, y_test = torch.max(y_test, dim=1)
            losses['val_scores'].append(loss_module(preds, y_test).cpu())

    device = torch.device('cuda')

    cifar10 = cifar10_utils.get_cifar10(f'cifar10/cifar-10-batches-py')
    train_data = cifar10_utils.DataSet(cifar10['train'].images, cifar10['train'].labels)
    test_data = cifar10_utils.DataSet(cifar10['test'].images, cifar10['test'].labels)

    def fit(**hyperparameter):
        model = torchvision.models.vgg13_bn(pretrained=True).to(device)
        classifier = list(model.classifier.children())
        classifier.pop()
        classifier.append(torch.nn.Linear(4096, 10))
        classifier = torch.nn.Sequential(*classifier).to(device)

        for param in model.parameters():
            param.requires_grad = False
        for param in list(model.parameters())[-5:]:
            param.requires_grad = True
        model.classifier = classifier

        loss_module = nn.CrossEntropyLoss()
        optimizer = hyperparameter['optimizer'](
            model.parameters(), lr=hyperparameter['learning_rate']
        )

        accuracies = dict(train_scores=list(), val_scores=list())
        losses = dict(train_scores=list(), val_scores=list())
        for i in range(FLAGS.max_steps):
            x, y = train_data.next_batch(FLAGS.batch_size)

            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)

            preds = model(x)
            preds = preds.squeeze(dim=1)

            if i % FLAGS.eval_freq == FLAGS.eval_freq - 1:
                if i % FLAGS.eval_freq == FLAGS.eval_freq - 1:
                    x_train, y_train = train_data.next_batch(2000)
                    x_test, y_test = test_data.next_batch(2000)
                    evaluate(model, x_train, y_train, x_test, y_test, loss_module, accuracies, losses)

            _, y = torch.max(y, dim=1)
            loss = loss_module(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plot_history(accuracies, "Accuracies", model_name="VGG19")
        plot_history(losses, "Losses", title="Train and Validation Losses of ", model_name="VGG19")

        # Test
        with torch.no_grad():
            x, y = test_data.next_batch(2000)

            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)

            preds = model(x)
            preds = preds.squeeze(dim=1)

            print("Test Accuracy: ", accuracy(preds, y))
            return accuracy(preds, y)

    fit(
        optimizer=torch.optim.Adam,
        learning_rate=FLAGS.learning_rate
    )
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
