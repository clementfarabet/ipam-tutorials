
Theano
======

The [Deep Learning Tutorials](http://www.deeplearning.net/tutorial/)(DLTs) provide
another perspective on how to do Deep Learning in Python. Rather than using
pyautodiff, which uses Theano internally, the DLTs are implemented using
Theano's API directly.


Using Theano's API makes it natural to express more general algorithms that do not fit neatly into the function-minimization framework.
Today we'll at Persistant Contrastive Divergence (PCD, also known as Stochastic Maximum Likelihood or SML).




Question idea PCD
- modify RBM to do PCD
- note that all the negative-phase particles are doing the same thing,
  which is a big waste of efficiency
- pre-condition PCD using N iterations of CD
- What is the effect of batchsize on convergence... any?


