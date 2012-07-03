
# Question 1 - which method produces the prettiest filters?
#    (Seriously, you might be surprised how often people diagnose whether
#    they're algorithm is working or not by just looking at the filters!)
#
#    Justify your choice with examples.


#import skdata.CIFAR10
import logging
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from skdata import mnist
import autodiff

from utils import tile_raster_images

import unsup


def show_filters(x, img_shape, tile_shape):
    out = tile_raster_images(x, img_shape, tile_shape, (1, 1))
    plt.imshow(out, cmap=plt.cm.gray)
    plt.show()


def main():
    # -- QQ feel free to change the number of hidden units to learn different
    # feature representations.
    n_hidden1 = n_hidden2 = 25
    n_hidden = n_hidden1 * n_hidden2

    # -- QQ compare float32 with float64
    # -- N.B. that you *need* float32 to run these optimizations on a GPU
    dtype = 'float32'

    rng = np.random.RandomState(123)
    n_examples = 10000

    data_view = mnist.views.OfficialVectorClassification(x_dtype=dtype)

    x = data_view.train.x[:n_examples]
    n_examples, n_visible = x.shape
    x_img_res = 28, 28

    # -- uncomment this line to see sample images from the data set
    # show_filters(x[:100], x_img_res, (10, 10))

    # -- allocate and initialize a model (w, visbias, hidbias)
    #    QQ - most/all of our filter-learning algorithms are sensitive to
    #         initial conditions.  How does the scale and range of the initial
    #         values affect the trajectory of learning?
    w = rng.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)).astype(dtype)
    visbias = np.zeros(n_visible).astype(dtype)
    hidbias = np.zeros(n_hidden).astype(dtype)

    # -- uncomment this line to visualize the initial filter bank:
    # show_filters(w.T, x_img_res, (n_hidden1, n_hidden2))

    online_batch_size = 1   # -- QQ: what happens when this is 2? 5? 10? 1000?

    x_stream = x.reshape((
        n_examples / online_batch_size,  # -- sgd will loop over this axis
        online_batch_size,
        x.shape[1]))

    # -- this inline function defines our feature-learning loss function for
    #    autodiff.
    def train_criterion(w, hidbias, visbias, x_i=x):
        # optional x_i parameter is used by the fmin_sgd's `stream` iterator
        # but not used by fmin_l_bfgs_b, so we make it default to the value
        # needed by fmin_l_bfgs_b.

        # -- QQ: try swapping in different feature-learning criteria here
        #        Can you interpret why the filters come out different or
        #        similar?

        # -- AUTO-ENCODER
        #cost, hid = unsup.logistic_autoencoder_binary_x(x_i, w, hidbias, visbias)

        # -- DENOISING AUTO-ENCODER
        cost, hid = unsup.denoising_autoencoder_binary_x(x_i, w, hidbias,
                visbias, noise_level=0.3)

        # -- RBM
        # cost, hid = unsup.rbm_binary_x(x_i, w, hidbias, visbias)

        # -- QQ: try different l1 and l2 penalties, what do they do? What
        # happens to the look of the filters when you mix them?
        l1_cost = abs(w).sum() * 0.0
        l2_cost = (w ** 2).sum() * 0.0
        return cost.mean() + l1_cost + l2_cost


    # -- ONLINE TRAINING
    for epoch in range(3):
        t0 = time.time()
        w, hidbias, visbias = autodiff.fmin_sgd(train_criterion,
                args=(w, hidbias, visbias),
                stream=x_stream,  # -- fmin_sgd will loop through this once
                stepsize=0.005,   # -- QQ: you should always tune this
                print_interval=1000,
                )

        # -- uncomment this line to visualize the post-online filter bank
        print 'Online training epoch %i took %f seconds' % (
                epoch, time.time() - t0)
        show_filters(w.T, x_img_res, (n_hidden1, n_hidden2))

    # -- BATCH TRAINING
    w, hidbias, visbias = autodiff.fmin_l_bfgs_b(train_criterion,
            args=(w, hidbias, visbias),
            # -- scipy.fmin_l_bfgs_b kwargs follow
            maxfun=100,   # -- how many iterations of BFGS to do
            iprint=1,     # -- 1 for verbose, 0 for normal, -1 for quiet
            m=20,         # -- how well to approximate the Hessian
            )

    # -- uncomment this line to visualize the post-batch filter bank
    show_filters(w.T, x_img_res, (n_hidden1, n_hidden2))


# -- this is the standard way to make a Python file both importable and
# executable
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sys.exit(main())
