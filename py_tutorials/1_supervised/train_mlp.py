import logging
import sys
import time

import numpy as np
from numpy import arange, dot, maximum, ones, tanh, zeros
from numpy.random import randn

from skdata import mnist
from autodiff import fmin_sgd, fmin_l_bfgs_b

from utils import show_filters


def svm(params, x, y1, n_classes=None):
    if 'svm' in params:
        w = params['svm']['weights']
        b = params['svm']['biases']
    else:
        w = zeros((x.shape[1], n_classes), dtype=x.dtype)
        b = zeros(n_classes, dtype=x.dtype)
        params['svm'] = {}
        params['svm']['weights'] = w
        params['svm']['biases'] = b


    margin = y1 * (dot(x, w) + b)
    hinge = np.maximum(0, 1 - margin)
    # -- one vs. all linear SVM loss
    cost = hinge.mean(axis=0).sum()
    return cost


def mlp_svm(params, x, y1, n_hiddens=None, n_classes=None):
    if 'mlp' in params:
        weights = params['mlp']['weights']
        biases = params['mlp']['biases']
    else:
        weights = []
        biases = []
        in_shape = x.shape[1]
        for n_hidden in n_hiddens:
            weights.append(randn(in_shape, n_hidden).astype(x.dtype) * 0.1)
            biases.append(zeros(n_hidden, dtype=x.dtype))
            in_shape = n_hidden
        params['mlp'] = {}
        params['mlp']['weights'] = weights
        params['mlp']['biases'] = biases

    hid = x
    for w, b in zip(weights, biases):
        hid = tanh(dot(hid, w) + b)

    cost = svm(params, hid, y1, n_classes)
    return cost


def main():
    # -- top-level parameters of this script
    dtype = 'float32'  # XXX
    n_examples = 50000
    online_batch_size = 1
    online_epochs = 2
    batch_epochs = 30
    lbfgs_m = 20
    n_mlp_hiddens = [200]  # -- one entry per hidden layer

    # -- load and prepare the data set
    data_view = mnist.views.OfficialVectorClassification(x_dtype=dtype)
    n_classes = 10
    x = data_view.train.x[:n_examples]
    y = data_view.train.y[:n_examples]
    y1 = -1 * ones((len(y), n_classes)).astype(dtype)
    y1[arange(len(y)), y] = 1

    # -- allocate the model by running one example through it
    init_params = {}
    mlp_svm(init_params, x[:1], y[:1], n_mlp_hiddens, n_classes)

    if online_epochs:
        # -- stage-1 optimization by stochastic gradient descent
        print 'Starting SGD'
        n_batches = n_examples / online_batch_size
        stage1_params, = fmin_sgd(mlp_svm, (init_params,),
                streams={
                    'x': x.reshape((n_batches, online_batch_size, x.shape[1])),
                    'y1': y1.reshape((n_batches, online_batch_size, y1.shape[1]))},
                loops=online_epochs,
                stepsize=0.001,
                print_interval=10000,
                )

        print 'SGD complete, about to start L-BFGS'
        show_filters(stage1_params['mlp']['weights'][0].T, (28, 28), (8, 25,))
    else:
        print 'Skipping stage-1 SGD'
        stage1_params = init_params

    # -- stage-2 optimization by L-BFGS
    if batch_epochs:
        def batch_mlp_svm(p):
            return mlp_svm(p, x, y1)

        print 'Starting L-BFGS'
        stage2_params, = fmin_l_bfgs_b(lambda p: mlp_svm(p, x, y1),
                args=(stage1_params,),
                maxfun=batch_epochs,
                iprint=1,
                m=lbfgs_m)

        print 'L-BFGS complete'
        show_filters(stage2_params['mlp']['weights'][0].T, (28, 28), (8, 25,))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sys.exit(main())

