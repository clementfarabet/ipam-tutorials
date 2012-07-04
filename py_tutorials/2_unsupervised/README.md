
### Question 2 - Hooking up a classifier.
    Step 1 - classify features with a linear SVM
    Step 2 - classify features with a multi-layer classifier
    Step 3 - realize you could do backprop through the whole system if you
             wanted to....
    Step 4 - consider - if your top layer features can achieve zero
             training error, might it still be interesting to do whole-system
             fine-tuning by backpropagation until that zero training error is
             reached?

### Which feature-learning method yields the lowest error?
   Think about all of the significant choices that go into this
   answer:
   - data set (encoding, nature of signal)
   - feature-learning hyperparameters
   - supervised learning algorithm on top
    (This isn't really a homework question - interest in this question is one
    of the major themes of the workshop)


### Question 1 - which method produces the prettiest filters?
    (Seriously, you might be surprised how often people diagnose whether
    they're algorithm is working or not by just looking at the filters!)

    Justify your choice with examples.


What happens to the filter-learning process when the noise-free autoencoder
is subjected to noise at various points in the algorithm calculation. All of
these might be justifiable in some sense as making the internal
representations more robust, why is applying noise to the input best? Is it?


## Question 2 - Hooking up a classifier.
    Step 1 - classify features with a linear SVM
    Step 2 - classify features with a multi-layer classifier
    Step 3 - realize you could do backprop through the whole system if you
             wanted to....
    Step 4 - consider - if your top layer features can achieve zero
             training error, might it still be interesting to do whole-system
             fine-tuning by backpropagation until that zero training error is
             reached?


## Which feature-learning method yields the lowest error?
#   Think about all of the significant choices that go into this
#   answer:
#   - data set (encoding, nature of signal)
#   - feature-learning hyperparameters
#   - supervised learning algorithm on top
#    (This isn't really a homework question - interest in this question is one
#    of the major themes of the workshop)


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



# Question 3 - Overcompleteness.
#   No matter how classes are distributed in a real vector space, it is always
#   possible to classify them with Bayes error by K-nearest neighbours, which
#   can be seen as a piecewise function approximator. A linear SVM is not a
#   piecewise function approximator though, and it needs at least as many
#   features as there are pieces.  One of the theoretical advantages of deep
#   architectures is that the pieces corresponding to a class can be
#   systematically merged, but however many pieces remain unmerged - that's
#   how many features we will need *as a minimum*  to get the right answers
#   from a linear classifier.

#   An *overcomplete* representation is typically used to denote `hid` vectors
#   with more dimensions than the `x` that it came from.
#
#   Try learning a linear autoencoder in an overcomplete regime, what happens?
#   (Hint, did your training amount to an identity matrix in `w`?)
#
#   Try learning a logistic autoencoder in an overcomplete regime with tied
#   weights, then with-untied weights... this will make a difference. Any idea
#   why?
#
#   Often an SVM classifier does better and better when a feature-learning
#   algorithm is used to produce more and more features, but with a
#   diminishing marginal return wrt the representation size. The shape of that
#   marginal utility curve is not always the same though! Recent work such as
#   [1] and [2] shows that simple high-dimensional features can actually
#   outperform more sophisticated features whose learning algorithms falter in
#   exteremely overcomplete regimes.
#
#   [1] Coates et al. XXX
#   [2] Pinto et al. XXX


# Question idea PCD
# - modify RBM to do PCD
# - note that all the negative-phase particles are doing the same thing,
#   which is a big waste of efficiency
# - pre-condition PCD using N iterations of CD
# - What is the effect of batchsize on convergence... any?

