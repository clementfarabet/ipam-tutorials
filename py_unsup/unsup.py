import numpy as np
from numpy import dot, exp, log, sqrt, tanh


"""

Standard Semantics
==================
x - observation matrix (#examples, #features)
x_rec - the reconstruction of `x`
hid - the features / hidden representation corresponding to `x`.
cost - the cost of each example in `x`

"""

#
# SECTION: Helper functions
#


def logictic(x):
    """Return logistic sigmoid of float or ndarray `x`"""
    return 1.0 / (1.0 + exp(-x))


def cross_entropy(x, x_rec_p):
    """Return the independent Bernoulli cross-entropy cost

    x_rec_p is the Bernoulli parameter of the model's reconstruction
    """
    # -- N.B. this is numerically bad, we're counting on Theano to fix up
    return (x * log(x_rec_p) + (1 - x) * log(1 - x_rec_p)).sum(axis=1)


def squared_error(x, x_rec):
    """Return the mean squared error of approximating `x` with `x_rec`"""
    return ((x - x_rec) ** 2).sum(axis=1)



#
# SECTION: Feature-learning criteria
#


# classic real-real autoencoder
def pca_autoencoder_real_x(x, w):
    hid = dot(x, w)
    x_rec = dot(hid, w.T)
    cost = squared_error_cost(x, x_rec)
    return cost, hid


# classic binary-binary autoencoder ("tied" weights)
def logistic_autoencoder_binary_x(x, w, hidbias, visbias)
    hid = logistic(dot(x, w) + hidbias)
    # -- using w.T here is called using "tied weights"
    # -- using a second weight matrix here is called "untied weights"
    x_rec = logistic(dot(hid, w.T) + visbias)
    cost = cross_entropy(x, x_rec)
    return cost, hid


# classic binary-binary RBM, Contastive Divergence (CD-1)
def rbm_binary_x(x, w, visbias, hidbias)
    hid = logistic(dot(x, w) + hidbias)
    hid_sample = hid > np.random.rand(*hid.shape)

    # -- N.B. model is not actually trained to reconstruct x
    x_rec = logistic(dot(hid_sample, w.T) + visbias)
    x_rec_sample = x_rec > np.random.rand(*x_rec.shape)

    # "negative phase" hidden unit expectation
    hid_rec = logistic(dot(x_rec_sample, w) + hidbias)

    def free_energy(x, h):
        return -log(h).sum(axis=1) - dot(x, visbias)

    cost = free_energy(x, hid) - free_energy(x_rec, hid_rec)
    return cost, hid


# classic real-discrete K-means
def k_means_real_x(x, w, hidbias, visbias):
    xw = dot(x, w) + hidbias
    # -- N.B. for so-called "soft" k-means convert xw into a smoother
    #    normalized distribution over the hidden states.
    hid = (xw == xw.max(axis=1)[:, newaxis])
    x_rec = dot(hid, w.T) + visbias
    cost = ((x - x_rec) ** 2).mean(axis=1)
    return cost, hid


# classic real-real Sparse Coding
def sparse_coding_real_x(x, w, sparse_coding_algo=FISTA):
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



