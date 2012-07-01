
# Question 1 - which method produces the prettiest filters?
#    (Seriously, you might be surprised how often people diagnose whether
#    they're algorithm is working or not by just looking at the filters!)
#
#    Justify your choice with examples.


#import skdata.CIFAR10
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

from skdata.mnist.views import OfficialVectorClassification
import autodiff

from utils import tile_raster_images

import unsup


def show_filters(x, img_shape, tile_shape):
    out = tile_raster_images(x, img_shape, tile_shape, (1, 1))
    plt.imshow(out, cmap=plt.cm.gray)
    plt.show()


def main():
    n_hidden = 16 * 16      # -- QQ feel free to change this
    dtype = 'float64'       # -- QQ compare float64?

    data_view = OfficialVectorClassification(x_dtype=dtype)

    x = data_view.train.x[:10000]
    n_examples, n_visible = x.shape

    # -- uncomment this line to see sample images from the data set
    # show_filters(x[:100], (28, 28), (10, 10))

    # -- allocate and initialize a model (w, visbias, hidbias)
    #    QQ - most/all of our filter-learning algorithms are sensitive to
    #         initial conditions.  How does the scale and range of the initial
    #         values affect the trajectory of learning?
    w = np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)).astype(dtype)
    visbias = np.zeros(n_visible).astype(dtype)
    hidbias = np.zeros(n_hidden).astype(dtype)

    # -- uncomment this line to visualize the initial filter bank
    # show_filters(w.T, (28, 28), (16, 16))

    if 0: # -- ONLINE TRAINING

        # -- sgd will loop over x_blocks' leading dimension:
        #                     ||
        #                     \/
        x_stream = x.reshape((10000, 1, x.shape[1]))

        def online_train_criterion(w, hidbias, visbias, x_i):
            cost, hid = unsup.logistic_autoencoder_binary_x(x_i, w, hidbias, visbias)
            return cost.mean()

        w, hidbias, visbias = autodiff.fmin_sgd(
                online_train_criterion,
                args=(w, hidbias, visbias),
                stream=x_stream,
                #stream_elements_have_same_shape=True, # XXX Implement this
                stepsize=0.01,
                )

        # -- uncomment this line to visualize the initial filter bank
        show_filters(w.T, (28, 28), (16, 16))

    if 1: # -- BATCH TRAINING
        def batch_train_criterion(w, hidbias, visbias):
            cost, hid = unsup.logistic_autoencoder_binary_x(x, w, hidbias, visbias)
            return cost.mean()

        w, hidbias, visbias = autodiff.fmin_l_bfgs_b(
                batch_train_criterion,
                args=(w, hidbias, visbias),
                # -- scipy.fmin_l_bfgs_b kwargs follow
                maxfun=10,
                iprint=1,
                )

        # XXX: WHY DOES L_BFGS SCREW UP THE FILTERS SO MUCH?!??

        # -- uncomment this line to visualize the initial filter bank
        show_filters(w.T, (28, 28), (16, 16))


# -- this is the standard way to make a Python file both importable and
# executable
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sys.exit(main())
