
from skdata import cifar10
n_examples = 10000
# -- RECOMMENDED: restart the IPython kernel to clear out memory
data_view = cifar10.view.OfficialImageClassification(x_dtype='float32', n_train=n_examples)
x = data_view.train.x[:n_examples].transpose(0, 3, 1, 2)


import numpy as np
from util import random_patches
from util import mean_and_std


remove_mean = True
hard_beta = False
beta = 1
gamma = 0.01

patches = random_patches(x, 1000, 7, 7, np.random)

def contrast_normalize(patches):
    X = patches
    if X.ndim != 2:
        raise TypeError('contrast_normalize requires flat patches')
    if remove_mean:
        xm = X.mean(1)
    else:
        xm = X[:,0] * 0
    Xc = X - xm[:, None]
    l2 = (Xc * Xc).sum(axis=1)
    if hard_beta:
        div2 = np.maximum(l2, beta)
    else:
        div2 = l2 + beta
    X = Xc / np.sqrt(div2[:, None])
    return X


def ZCA_whiten(patches):
    # -- ZCA whitening (with band-pass)

    # Algorithm from Coates' sc_vq_demo.m

    X = patches.reshape(len(patches), -1).astype('float64')

    X = contrast_normalize(X)
    print 'patch_whitening_filterbank_X starting ZCA'
    M, _std = mean_and_std(X)
    Xm = X - M
    assert Xm.shape == X.shape
    print 'patch_whitening_filterbank_X starting ZCA: dot', Xm.shape
    C = np.dot(Xm.T, Xm) / (Xm.shape[0] - 1)
    print 'patch_whitening_filterbank_X starting ZCA: eigh'
    D, V = np.linalg.eigh(C)
    print 'patch_whitening_filterbank_X starting ZCA: dot', V.shape
    P = np.dot(np.sqrt(1.0 / (D + gamma)) * V, V.T)
    assert M.ndim == 1
    return M, P, X

MPX = ZCA_whiten(patches)

def fb_whitened_projections(patches, MPX, n_filters, rseed, dtype):
    """
    MPX is the output of ZCA_whiten

    M, and fb will be reshaped to match elements of patches
    """
    M, P, patches_cn = MPX
    if patches_cn.ndim != 2:
        raise TypeError('wrong shape for MPX args, should be flattened',
                patches_cn.shape)
    rng = np.random.RandomState(rseed)
    D = rng.randn(n_filters, patches_cn.shape[1])
    D = D / (np.sqrt((D ** 2).sum(axis=1))[:, None] + 1e-20)
    fb = np.dot(D, P)
    fb = fb.reshape((n_filters,) + patches.shape[1:])
    M = M.reshape(patches.shape[1:])
    M = M.astype(dtype)
    fb = fb.astype(dtype)
    if fb.size == 0:
        raise ValueError('filterbank had size 0')
    return M, fb


fb_whitened_projections(patches, MPX, 10, 123, 'float32')

def fb_whitened_patches(patches, MPX, n_filters, rseed, dtype):
    """
    MPX is the output of patch_whitening_filterbank_X with reshape=False

    M, and fb will be reshaped to match elements of patches

    """
    M, P, patches_cn = MPX
    print M.shape, P.shape, patches_cn.shape
    rng = np.random.RandomState(rseed)
    d_elems = rng.randint(len(patches_cn), size=n_filters)
    D = np.dot(patches_cn[d_elems] - M, P)
    D = D / (np.sqrt((D ** 2).sum(axis=1))[:, None] + 1e-20)
    fb = np.dot(D, P)
    fb.shape = (n_filters,) + patches.shape[1:]
    M.shape = patches.shape[1:]
    M = M.astype(dtype)
    fb = fb.astype(dtype)
    if fb.size == 0:
        raise ValueError('filterbank had size 0')
    return M, fb

fb_whitened_patches(patches, MPX, 10, 123, 'float32')
