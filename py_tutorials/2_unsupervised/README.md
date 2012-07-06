Unsupervised Feature Learning in Python
=======================================

Despite philosophical differences which may some times be over-emphasized, there
are a lot of algorithmic and mathematical commonalities among the unsupervised
algorithms used so far for deep learning.
I have written up a little demonstration script
[learn_features.py](./2_unsupervised/learn_features.py)
to illustrate what I mean.  The script allows you to swap in and out
feature-learning approaches that have been proposed:

* Auto-Encoder (AE)
* K-means
* De-noising Auto-Encoder (DAE)
* PCA (admittedly, a non-standard "de-normalized and un-sorted" PCA if you will)
* Restricted Boltzmann Machine (RBM)
* Sparse Coding


For a quick introduction to the theory and background of the autoencoder models
and the RBM, I encourage you to read through the
[Deep Learning Tutorials][dlt].
The main difference between those tutorials and this one is that they are built
around more purely Theano implementations.

The [learn_features.py][lf] script is meant to
be called from the command line with no arguments, and it will print out some
numbers and show pictures of the learned model from time to time.

    $ python learn_features.py

Open up the file and modify some of the constants below where it says `def
main():`  to get started at changing the program logic.  The rest of this file
is an enumeration of the different ways that you might want to modify that
script to explore the various feature-learning algorithms.


## Learning Filters

Try replacing the line that starts with `algo = ` so that different cost
functions are used by the `train_criterion`.
* Which of the methods produces filters that are more similar?
* Which method produces the prettiest filters?
* Why do they look the way they do?
  (People often diagnose whether an algorithm is working just by 
  looking at the filters!)
* How is the rate and nature of the learning process affected by:
    - learning rate
    - L1 regularization
    - L2 regularization
    - amount of noise (when applicable)
    - number of hidden units


## Make up a new Algorithm!

If you look across the various cost functions in [learn_features.py][lf]
you can see that there are common ingredients being combined in different ways.
What happens when you mix them up?
Can you think of reasons for why these hybrids might
even be theoretically justified?

Here are some ideas to get you started:

```python

def rbf_autoencoder(x, w, hidbias, visbias)
    hid = logistic(euclidean_distances(x-visbias, w.T, squared=True) + hidbias)
    x_reconstruction = dot(hid, w.T) + visbias
    cost = ((x - x_rec) ** 2).mean(axis=1)
    return cost, hid

def soft_k_means(x, w, hidbias, visbias):
    xw = euclidean_distances2(x - visbias, w.T)
    # -- This calculates a soft winner
    hid = softmax(-xw)
    x_rec = dot(hid, w.T)
    cost = ((x - x_rec) ** 2).mean(axis=1)
    return cost, hid


```

What would a de-noising rbf autoencoder look like? Would it work?
Can you think of a faster way to convert `xw` to a distribution over hidden
states `hid` in the `soft_k_means` algorithm above? Would it be any worse
than the softmax or hard k-means strategy?


## Classifying features with an SVM

Often people wonder which feature-learning method yields the lowest error.
Think about all of the decisions that go into this answer, such as:
- data set (nature of signal, pre-processing)
- feature-learning hyperparameters
- supervised learning algorithm on top


XXX

- classify features with a linear SVM
- classify features with a multi-layer classifier
- realize you could do backprop through the whole system if you wanted to....
- consider - if your top layer features can achieve zero
     training error, might it still be interesting to do whole-system
     fine-tuning by backpropagation until that zero training error is
     reached?

### Classification and Overcompleteness

No matter how classes are distributed in a real vector space, it is always
possible to classify them with Bayes error by K-nearest neighbours, which
can be seen as a piecewise function approximator. A linear SVM is not a
piecewise function approximator though, and it needs at least as many
features as there are pieces.  One of the theoretical advantages of deep
architectures is that the pieces corresponding to a class can be
systematically merged, but however many pieces remain unmerged - that is
how many features we will need *as a minimum*  to get the right answers
from a linear classifier.

An *overcomplete* representation is typically used to denote `hid` vectors
with more dimensions than the `x` that it came from.

Try learning a linear autoencoder in an overcomplete regime, what happens?
(Hint, did your training amount to an identity matrix in `w`?)

Try learning a logistic autoencoder in an overcomplete regime with tied
weights, then with-untied weights... this will make a difference. Any idea
why?

Often an SVM classifier does better and better when a feature-learning
algorithm is used to produce more and more features, but with a
diminishing marginal return wrt the representation size. The shape of that
marginal utility curve is not always the same though! Recent work such as
[coates] and [pinto] shows that simple high-dimensional features can actually
outperform more sophisticated features whose learning algorithms falter in
exteremely overcomplete regimes.

[lf]: ./2_unsupervised/learn_features.py "learn_features.py"
[dlt]: http://deeplearning.net/tutorial/ "Deep Learning Tutorials"
[coates] XXX "Coates et al."
[pinto] XXX "Pinto et al."


## Deep Learning

XXX Don't have a way to visualize these very well (try gradient ascent on
activation? Spike-triggered Average?)


## Drawing Samples from RBM

XXX


