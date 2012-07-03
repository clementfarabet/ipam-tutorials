
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

