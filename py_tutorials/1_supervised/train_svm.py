import sys
import time

import numpy as np
from numpy import arange, dot, maximum, ones, tanh, zeros
from numpy.random import randn

from skdata import mnist
from autodiff import fmin_sgd, fmin_l_bfgs_b

from utils import show_filters


# -- top-level parameters of this script
n_conv_hiddens = [(16, (5, 5)), (64, 5, 5)]
n_mlp_hiddens = [200]
n_classes = 10
dtype = 'float32'  # XXX
n_examples = 50000
online_batch_size = 1
online_epochs = 2
batch_epochs = 30
lbfgs_m = 20


# -- load and prepare the data set
data_view = mnist.views.OfficialVectorClassification(x_dtype=dtype)
x = data_view.train.x[:n_examples]
y = data_view.train.y[:n_examples]
y1 = -1 * ones((len(y), n_classes))
y1[arange(len(y)), y] = 1

# --initialize the SVM model
w = zeros((x.shape[1], n_classes))
b = zeros(n_classes)

def svm(ww, bb, xx=x, yy=y1):
    margin = yy * (dot(xx, ww) + bb)
    hinge = maximum(0, 1 - margin)
    # -- one vs. all linear SVM loss
    cost = hinge.mean(axis=0).sum()
    return cost

# -- stage-1 optimization by stochastic gradient descent
print 'Starting SGD'
n_batches = n_examples / online_batch_size
w, b = fmin_sgd(svm, (w, b),
        streams={
            'xx': x.reshape((n_batches, online_batch_size, x.shape[1])),
            'yy': y1.reshape((n_batches, online_batch_size, y1.shape[1]))},
        loops=online_epochs,
        stepsize=0.000001,
        print_interval=10000,
        )

print 'SGD complete, about to start L-BFGS'
show_filters(w.T, (28, 28), (2, 5,))

# -- stage-2 optimization by L-BFGS
print 'Starting L-BFGS'
w, b = fmin_l_bfgs_b(svm, (w, b),
        maxfun=batch_epochs,
        iprint=1,
        m=lbfgs_m)

print 'L-BFGS complete'
show_filters(w.T, (28, 28), (2, 5,))

