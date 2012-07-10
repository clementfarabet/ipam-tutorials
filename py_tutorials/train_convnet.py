import logging
import sys
import time

import numpy as np
from numpy import arange, dot, maximum, ones, tanh, zeros
from numpy.random import randn

from skdata import mnist
from autodiff import fmin_sgd, fmin_l_bfgs_b

from util import hinge
from util import ova_svm_prediction, ova_svm_cost
from util import mlp_prediction, mlp_cost
from util import fbncc  # -- filterbank normalized cross-correlation

def tanh_conv_layer(W_fb, b_fb, imgs):
    # -- similar to tanh_layer, but we used fbncc instead of dot
    return np.tanh(fbncc(imgs, W_fbncc) + b_fb)


# -- top-level parameters of this script
dtype = 'float32'  # helps save memory and go faster
n_examples = 10000
online_batch_size = 1

n_hidden

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

