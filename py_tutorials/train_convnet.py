import logging
import sys
import time

import numpy as np
from numpy import arange, dot, maximum, ones, tanh, zeros
from numpy.random import uniform

from skdata import mnist
import autodiff

from util import hinge
from util import ova_svm_prediction, ova_svm_cost
from util import tanh_layer
from util import mlp_prediction, mlp_cost

# -- filterbank normalized cross-correlation
from util import fbncc

# -- max-pooling over 2x2 windows
from util import max_pool_2d_2x2

# -- load and prepare the data set
#
# -- N.B. we're loading up x as images this time, not vectors
dtype = 'float32'  # helps save memory and go faster
n_examples = 10000
data_view = mnist.views.OfficialImageClassification(x_dtype=dtype)
n_classes = 10
x_rows, x_cols = data_view.train.x.shape[1:3]
# -- N.B. we shuffle the input to have shape
#    (#examples, #channels, #rows, #cols)
x = data_view.train.x[:n_examples].transpose(0, 3, 1, 2)

y = data_view.train.y[:n_examples]
y1 = -1 * ones((len(y), n_classes)).astype(dtype)
y1[arange(len(y)), y] = 1


# -- similar to tanh_layer, but we used fbncc instead of dot
def tanh_conv_layer(W_fb, b_fb, img4):
    activation = fbncc(img4, W_fb) + b_fb
    activation = max_pool_2d_2x2(activation)
    return np.tanh(activation)


# -- top-level parameters of this script

n_filters = 16
patch_height = 5
patch_width = 5
patch_size = patch_height * patch_width

# -- allocate one convolutional layer
W_fb = uniform(
        low=-np.sqrt(6.0 / (patch_size + n_filters)),
        high=np.sqrt(6.0 / (patch_size + n_filters)),
        size=(n_filters, 1, patch_height, patch_width),
        ).astype(x.dtype)
b_fb = zeros((
            n_filters,
            x_rows - patch_height + 1,
            x_cols - patch_width + 1),
        dtype=x.dtype)

# initialize input layer parameters
n_hidden = 200

V = uniform(low=-np.sqrt(6.0 / (b_fb.size // 4 + n_hidden)),
                high=np.sqrt(6.0 / (b_fb.size // 4 + n_hidden)),
                size=(b_fb.size // 4, n_hidden)).astype(x.dtype)
c = zeros(n_hidden, dtype=x.dtype)

# now allocate the SVM at the top
W = zeros((n_hidden, n_classes), dtype=x.dtype)
b = zeros(n_classes, dtype=x.dtype)

def convnet_prediction(W_fb, b_fb, V, c, W, b, x):
    layer1 = tanh_conv_layer(W_fb, b_fb, x)
    layer1_size = np.prod(layer1.shape[1:])
    layer2 = tanh_layer(V, c,
            np.reshape(layer1, (x.shape[0], layer1_size)))
    prediction = ova_svm_prediction(W, b, layer2)
    return prediction

def convnet_cost(W_fb, b_fb, V, c, W, b, x, y1):
    layer1 = tanh_conv_layer(W_fb, b_fb, x)
    layer1_size = np.prod(layer1.shape[1:])
    layer2 = tanh_layer(V, c,
            np.reshape(layer1, (x.shape[0], layer1_size)))
    cost = ova_svm_cost(W, b, layer2, y1)
    return cost

print convnet_cost(W_fb, b_fb, V, c, W, b, x[:3], y1[:3])

online_batch_size = 1
n_batches = n_examples / online_batch_size
W_fb, b_fb, V, c, W, b = autodiff.fmin_sgd(convnet_cost, (W_fb, b_fb, V, c, W, b),
            streams={
                'x': x.reshape((n_batches, online_batch_size,) + x.shape[1:]),
                'y1': y1.reshape((n_batches, online_batch_size, y1.shape[1]))},
            loops=5,
            stepsize=0.01,
            print_interval=1000,
            )
print 'SGD took %.2f seconds' % (time.time() - t0)

