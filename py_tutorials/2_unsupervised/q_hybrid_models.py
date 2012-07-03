
# Hybrid between K-means and logistic auto-encoder
def rbf_autoencoder_real_x(x, w, hidbias, visbias)
    hid = tanh(dot(x, w) + hidbias)
    x_reconstruction = dot(hid, w.T) + visbias
    cost = ((x - x_rec) ** 2).mean(axis=1)
    return cost, hid


# Hybrid between sparse coding and K-means:
def cosine_k_means_real_x(x, w):
    xw = dot(x, w) / sqrt
    # -- N.B. for so-called "soft" k-means convert xw into a smoother
    #    normalized distribution over the hidden states.
    hid = (xw == xw.max(axis=1)[:, newaxis])
    x_rec = dot(hid, w.T) + visbias
    cost = ((x - x_rec) ** 2).mean(axis=1)
    return cost, hid


