"""
Script that demonstrates multiple feature-learning algorithms

The script lacks a command-line interface, just modify the values at the top of the main()
function in-place.

"""

import logging
import sys
import time

import numpy as np

from skdata import mnist, cifar10, streetsigns
import autodiff

from utils import show_filters
import unsup


def main():
    # -- top-level parameters of this script
    n_hidden1 = n_hidden2 = 25
    dtype = 'float32'
    n_examples = 10000
    online_batch_size = 1
    online_epochs = 3
    algo = 'autoencoder'  # -- see below for the other options
    # -- parameters for lbfgs
    batch_epochs = 10
    lbfgs_m = 20
    dae_noise = 0.3

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

    def train_criterion(w, hidbias, visbias, x_i=x):
        if algo == 'autoencoder':
            cost, hid = unsup.logistic_autoencoder_binary_x(
                    x_i, w, hidbias, visbias)
        elif algo == 'dae':
            cost, hid = unsup.denoising_autoencoder_binary_x(
                    x_i, w, hidbias, visbias, noise_level=dae_noise)
        elif algo == 'rbm':
            cost, hid = unsup.rbm_binary_x(
                    x_i, w, hidbias, visbias)
        else:
            # -- there is a pca, and a k-means implementation
            #    in unsup as well to patch in here.
            raise NotImplementedError('unrecognized algo', algo)

        l1_cost = abs(w).sum() * 0.0
        l2_cost = (w ** 2).sum() * 0.0
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
