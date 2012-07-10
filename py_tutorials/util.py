""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy as np
import matplotlib.pyplot as plt

import theano
from theano import tensor
from theano.tensor.nnet.conv import conv2d


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


def show_filters(x, img_shape, tile_shape):
    """
    Call matplotlib imshow on the rows of `x`, interpreted as images.

    Parameters:
    x          - a matrix with T rows of P columns
    img_shape  - a (height, width) pair such that `height * width == P`
    tile_shape - a (rows, cols) pair such that `rows * cols == T`
    """
    out = tile_raster_images(x, img_shape, tile_shape, (1, 1))
    plt.imshow(out, cmap=plt.cm.gray, interpolation='nearest')
#    plt.show()


def hinge(margin):
    """Return elementwise hinge loss of margin ndarray"""
    return np.maximum(0, 1 - margin)


def ova_svm_prediction(W, b, x):
    """
    Return a vector of M integer predictions

    Parameters:
    W : weight matrix of shape (N, L)
    b : bias vector of shape (L,)
    x : feature vector of shape (M, N)
    """
    return np.argmax(np.dot(x, W) + b, axis=1)


def ova_svm_cost(W, b, x, y1):
    """
    Return a vector of M example costs using hinge loss

    Parameters:
    W : weight matrix of shape (N, L)
    b : bias vector of shape (L,)
    x : feature vector of shape (M, N)
    y1: +-1 labels matrix shape (M, L)
    """
    # -- one vs. all linear SVM loss
    margin = y1 * (np.dot(x, W) + b)
    cost = hinge(margin).mean(axis=0).sum()
    return cost


def tanh_layer(V, c, x):
    """
    Return layer output matrix of shape (#examples, #outputs)

    Parameters:
    V : weight matrix of shape (#inputs, #outputs)
    c : bias vector of shape (#outputs,)
    x : feature matrix of shape (#examples, #inputs)
    """
    return np.tanh(np.dot(x, V) + c)


def mlp_prediction(V, c, W, b, x):
    h = tanh_layer(V, c, x)
    return ova_svm_prediction(W, b, h)


def mlp_cost(V, c, W, b, x, y1):
    h = tanh_layer(V, c, x)
    return ova_svm_cost(W, b, h, y1)


def theano_fbncc(img4, img_shp, filters4, filters4_shp,
        remove_mean=True, beta=1e-8, hard_beta=True,
        shift3=0, dtype=None, ret_shape=False):
    """
    Channel-major filterbank normalized cross-correlation

    For each valid-mode patch (p) of the image (x), this transform computes

    p_c = (p - mean(p)) if (remove_mean) else (p)
    qA = p_c / sqrt(var(p_c) + beta)           # -- Coates' sc_vq_demo
    qB = p_c / sqrt(max(sum(p_c ** 2), beta))  # -- Pinto's lnorm

    There are two differences between qA and qB:

    1. the denominator contains either addition or max

    2. the denominator contains either var or sum of squares

    The first difference corresponds to the hard_beta parameter.
    The second difference amounts to a decision about the scaling of the
    output, because for every function qA(beta_A) there is a function
    qB(betaB) that is identical, except for a multiplicative factor of
    sqrt(N - 1).

    I think that in the context of stacked models, the factor of sqrt(N-1) is
    undesirable because we want the dynamic range of all outputs to be as
    similar as possible. So this function implements qB.

    Coates' denominator had var(p_c) + 10, so what should the equivalent here
    be?
    p_c / sqrt(var(p_c) + 10)
    = p_c / sqrt(sum(p_c ** 2) / (108 - 1) + 10)
    = p_c / sqrt((sum(p_c ** 2) + 107 * 10) / 107)
    = sqrt(107) * p_c / sqrt((sum(p_c ** 2) + 107 * 10))

    So Coates' pre-processing has beta = 1070, hard_beta=False. This function
    returns a result that is sqrt(107) ~= 10 times smaller than the Coates
    whitening step.

    """
    if dtype is None:
        dtype = img4.dtype

    # -- kernel Number, Features, Rows, Cols
    kN, kF, kR, kC = filters4_shp

    # -- patch-wise sums and sums-of-squares
    box_shp = (1, kF, kR, kC)
    box = tensor.addbroadcast(theano.shared(np.ones(box_shp, dtype=dtype)), 0)
    p_sum = conv2d(img4, box, image_shape=img_shp, filter_shape=box_shp)
    p_mean = 0 if remove_mean else p_sum / (kF * kR * kC)
    p_ssq = conv2d(img4 ** 2, box, image_shape=img_shp, filter_shape=box_shp)

    # -- this is an important variable in the math above, but
    #    it is not directly used in the fused lnorm_fbcorr
    # p_c = x[:, :, xs - xs_inc:-xs, ys - ys_inc:-ys] - p_mean

    # -- adjust the sum of squares to reflect remove_mean
    p_c_sq = p_ssq - (p_mean ** 2) * (kF * kR * kC)
    if hard_beta:
        p_div2 = tensor.maximum(p_c_sq, beta)
    else:
        p_div2 = p_c_sq + beta

    p_scale = 1.0 / tensor.sqrt(p_div2)

    # --
    # from whitening, we have a shift and linear transform (P)
    # for each patch (as vector).
    #
    # let m be the vector [m m m m] that replicates p_mean
    # let a be the scalar p_scale
    # let x be an image patch from s_imgs
    #
    # Whitening means applying the affine transformation
    #   (c - M) P
    # to contrast-normalized patch c = a (x - m),
    # where a = p_scale and m = p_mean.
    #
    # We also want to extract features in dictionary
    #
    #   (c - M) P
    #   = (a (x - [m,m,m]) - M) P
    #   = (a x - a [m,m,m] - M) P
    #   = a x P - a [m,m,m] P - M P
    #

    Px = conv2d(img4, filters4[:, :, ::-1, ::-1],
            image_shape=img_shp,
            filter_shape=filters4_shp,
            border_mode='valid')

    s_P_sum = filters4.sum(axis=[1, 2, 3])
    Pmmm = p_mean * s_P_sum.dimshuffle(0, 'x', 'x')
    if shift3:
        s_PM = (shift3 * filters4).sum(axis=[1, 2, 3])
        z = p_scale * (Px - Pmmm) - s_PM.dimshuffle(0, 'x', 'x')
    else:
        z = p_scale * (Px - Pmmm)

    assert z.dtype == img4.dtype
    z_shp = (img_shp[0], kN, img_shp[2] - kR + 1, img_shp[3] - kC + 1)
    if ret_shape:
        return z, z_shp
    else:
        return z


_fbncc_cache = {}
def fbncc(img4, kern4):
    """
    Return filterbank normalized cross-correlation

    Output has shape
    (#images, #kernels, #rows - #height + 1, #cols - #width + 1)

    See `theano_fbncc` for full documentation of transform.

    Parameters:
    img4 - images tensor of shape (#images, #channels, #rows, #cols)
    kern4 - kernels tensor of shape (#kernels, #channels, #height, #width)

    """
    assert img4.ndim == kern4.ndim == 4
    key = img4.shape + kern4.shape + (img4.dtype, kern4.dtype)
    if key not in _fbncc_cache:
        s_i = theano.tensor.tensor(dtype=img4.dtype, broadcastable=(1, 1, 0, 0))
        s_k = theano.tensor.tensor(dtype=kern4.dtype, broadcastable=(0, 1, 0, 0))
        s_y = theano_fbncc(
                s_i, img4.shape,
                s_k, kern4.shape,
                )

        f = theano.function([s_i, s_k], s_y)
        _fbncc_cache[key] = f
    else:
        f = _fbncc_cache[key]
    return f(img4, kern4)


def max_pool(img4):
    raise NotImplementedError()

