"""
Script that demonstrates multiple feature-learning algorithms

The script lacks a command-line interface, just modify the values at the top of the main()
function in-place.

Standard Semantics
==================
x - observation matrix (#examples, #features)
x_rec - the reconstruction of `x`
w - the weight matrix (#features, #hidden), aka dictionary
hid - the features / hidden representation corresponding to `x`.
cost - the cost of each example in `x`

"""

from functools import partial
import logging
import sys
import time

import numpy as np
from numpy import dot, exp, log, newaxis, sqrt, tanh
from numpy.random import rand

from skdata import mnist, cifar10, streetsigns
import autodiff

from utils import show_filters


#
# SECTION: Helper functions
#

def euclidean_distances2(X, Y):
    """Return all-pairs squared distances between rows of X and Y
    """
    # N.B. sklearn.metrics.pairwise.euclidean_distances
    # offers a more robust version of this routine,
    # but which does things that autodiff currently does not support.
    XX = np.sum(X * X, axis=1)[:, newaxis]
    YY = np.sum(Y * Y, axis=1)[newaxis, :]
    distances = XX - 2 * dot(X, Y.T) + YY
    np.maximum(distances, 0, distances)
    return distances


def cross_entropy(x, x_rec_p):
    """Return the independent Bernoulli cross-entropy cost

    x_rec_p is the Bernoulli parameter of the model's reconstruction
    """
    # -- N.B. this is numerically bad, we're counting on Theano to fix up
    return -(x * log(x_rec_p) + (1 - x) * log(1 - x_rec_p)).sum(axis=1)


def logistic(x):
    """Return logistic sigmoid of float or ndarray `x`"""
    return 1.0 / (1.0 + exp(-x))


def softmax(x):
    """Calculate the softmax of each row in x"""
    x2 = x - x.max(axis=1)[:, newaxis]
    ex = exp(x2)
    return ex / ex.sum(axis=1)[:, newaxis]


def squared_error(x, x_rec):
    """Return the mean squared error of approximating `x` with `x_rec`"""
    return ((x - x_rec) ** 2).sum(axis=1)


#
# SECTION: Feature-learning criteria
#


# real-real autoencoder
def pca_autoencoder_real_x(x, w, hidbias, visbias):
    hid = dot(x - visbias, w)
    x_rec = dot(hid, w.T)
    cost = squared_error(x - visbias, x_rec)
    return cost, hid


# binary-binary autoencoder (with "tied" weights)
def logistic_autoencoder_binary_x(x, w, hidbias, visbias):
    hid = logistic(dot(x, w) + hidbias)
    # -- using w.T here is called using "tied weights"
    # -- using a second weight matrix here is called "untied weights"
    x_rec = logistic(dot(hid, w.T) + visbias)
    cost = cross_entropy(x, x_rec)
    return cost, hid


# De-Noising binary-binary autoencoder (again, with "tied" weights)
def denoising_autoencoder_binary_x(x, w, hidbias, visbias, noise_level):
    # -- corrupt the input by zero-ing out some values randomly
    noisy_x = x * (rand(*x.shape) > noise_level)
    hid = logistic(dot(noisy_x, w) + hidbias)
    # -- using w.T here is called using "tied weights"
    # -- using a second weight matrix here is called "untied weights"
    x_rec = logistic(dot(hid, w.T) + visbias)
    cost = cross_entropy(x, x_rec)
    return cost, hid


# binary-binary RBM, Contastive Divergence (CD-1)
def rbm_binary_x(x, w, hidbias, visbias):
    hid = logistic(dot(x, w) + hidbias)
    hid_sample = (hid > rand(*hid.shape)).astype(x.dtype)

    # -- N.B. model is not actually trained to reconstruct x
    x_rec = logistic(dot(hid_sample, w.T) + visbias)
    x_rec_sample = (x_rec > rand(*x_rec.shape)).astype(x.dtype)

    # "negative phase" hidden unit expectation
    hid_rec = logistic(dot(x_rec_sample, w) + hidbias)

    def free_energy(xx):
        xw_b = dot(xx, w) + hidbias
        return -log(1 + exp(xw_b)).sum(axis=1) - dot(xx, visbias)

    cost = free_energy(x) - free_energy(x_rec_sample)
    return cost, hid


# real-discrete K-means
def k_means_real_x(x, w, hidbias, visbias):
    xw = euclidean_distances2(x - visbias, w.T)
    # -- This calculates a hard winner
    hid = (xw == xw.min(axis=1)[:, newaxis])
    x_rec = dot(hid, w.T)
    cost = ((x - x_rec) ** 2).mean(axis=1)
    return cost, hid


FISTA = NotImplementedError
# real-real Sparse Coding
def sparse_coding_real_x(x, w, hidbias, visbias, sparse_coding_algo=FISTA):
    # -- several sparse coding algorithms have been proposed, but they all
    # give rise to a feature learning algorithm that looks like this:
    hid = sparse_coding_algo(x, w)
    x_rec = dot(hid, w.T) + visbias
    cost = ((x - x_rec) ** 2).mean(axis=1)
    # -- the gradient on this cost wrt `w` through the sparse_coding_algo is
    # often ignored. At least one notable exception is the work of Karol
    # Greggor.  I feel like the Implicit Differentiation work of Drew Bagnell
    # is another, but I'm not sure.
    return cost, hid


#
# SECTION MAIN ROUTINE DRIVER
#

def main():
    # -- top-level parameters of this script
    n_hidden1 = n_hidden2 = 25
    dtype = 'float32'
    n_examples = 10000
    online_batch_size = 1
    online_epochs = 3

    # -- TIP: partial creates a new function with some parameters filled in
    # algo = partial(denoising_autoencoder_binary_x, noise_level=0.3)
    algo = logistic_autoencoder_binary_x

    batch_epochs = 10
    lbfgs_m = 20

    n_hidden = n_hidden1 * n_hidden2
    rng = np.random.RandomState(123)

    data_view = mnist.views.OfficialVectorClassification(x_dtype=dtype)
    x = data_view.train.x[:n_examples]
    n_examples, n_visible = x.shape
    x_img_res = 28, 28

    # -- uncomment this line to see sample images from the data set
    # show_filters(x[:100], x_img_res, (10, 10))

    # -- create a new model  (w, visbias, hidbias)
    w = rng.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)).astype(dtype)
    visbias = np.zeros(n_visible).astype(dtype)
    hidbias = np.zeros(n_hidden).astype(dtype)

    # show_filters(w.T, x_img_res, (n_hidden1, n_hidden2))
    x_stream = x.reshape((
        n_examples / online_batch_size,
        online_batch_size,
        x.shape[1]))

    def train_criterion(ww, hbias, vbias, x_i=x):
        cost, hid = algo(x_i, ww, hbias, vbias)
        l1_cost = abs(ww).sum() * 0.0    # -- raise 0.0 to enforce l1 penalty
        l2_cost = (ww ** 2).sum() * 0.0  # -- raise 0.0 to enforce l2 penalty
        return cost.mean() + l1_cost + l2_cost

    # -- ONLINE TRAINING
    for epoch in range(online_epochs):
        t0 = time.time()
        w, hidbias, visbias = autodiff.fmin_sgd(train_criterion,
                args=(w, hidbias, visbias),
                stream=x_stream,  # -- fmin_sgd will loop through this once
                stepsize=0.005,   # -- QQ: you should always tune this
                print_interval=1000,
                )
        print 'Online training epoch %i took %f seconds' % (
                epoch, time.time() - t0)
        show_filters(w.T, x_img_res, (n_hidden1, n_hidden2))

    # -- BATCH TRAINING
    w, hidbias, visbias = autodiff.fmin_l_bfgs_b(train_criterion,
            args=(w, hidbias, visbias),
            # -- scipy.fmin_l_bfgs_b kwargs follow
            maxfun=batch_epochs,
            iprint=1,     # -- 1 for verbose, 0 for normal, -1 for quiet
            m=lbfgs_m,         # -- how well to approximate the Hessian
            )

    show_filters(w.T, x_img_res, (n_hidden1, n_hidden2))


# -- this is the standard way to make a Python file both importable and
# executable
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sys.exit(main())
