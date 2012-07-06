# IPAM Graduate Summer School

On Deep Learning, Feature Learning
July 9 - 27, 2012

[More info here.](http://www.ipam.ucla.edu/programs/gss2012/) 


## Day 1: Setup

* 5 mins rapid overview of how we're going to organize the practical
  sessions. What are our goals for students, what we're offering to show.

* 10 mins crash course in Python, numpy, Theano

* 10 mins crash course in Lua, Torch7
  (will review very basic Lua and Torch concepts, to get people started)
  (will also do some basic review of available ML packages, graphical models, and so on)

* 15 min Getting people into groups and setting them up to run the sample code
  on laptop or EC2. Once they get it running, they can go for lunch or stick
  around and play with things.


## Day 2: Supervised Learning

* Train a ConvNet on multiple datasets

  * Q0. Loading/processing datasets is trivial. We have MNIST,
        CIFAR, and Google House Numbers available. The latter
        is the most exciting, because very few results are available
        at this time (and is more computer visionny that MNIST)

  * Q1. SGD preconditioning (number of updates, batchsize)

  * Q2. L-BFGS / ASGD optimization (number of updates, batchsize)
        Possibly show that SGD (or ASGD) is still the best method to
        obtain most general results when doing purely supervised learning

  * Debugging techniques?


## Day 3: Greedy Feature Learning

* Random, Imprinting, PCA, ZCA, K-Means, Autoencoder, RBM, Sparse coding, PSD Autoencoder (Torch)

  * Design sample code to show how similar these algorithms are

  * Vary the number of features?

  * Classify them with SVM?

  * Try to out-perform the supervised approach from Day 2.


## Day 4: Model Selection

* Talk about parameterizing your experiments to work with a database (MongoDB)
  and using Hyperparameter Optimization Algorithms (Grid, Random, TPE, GP,
  SMAC).

* Open time for questions?

* Assignment idea - figuring out which hyper-parameters are important. In
  prep - run the code from Day 2/3 for a while to produce a database of
  results. How good is the best model? How much spread is there among the best
  models? Is this a reliable max? What can you discern about the parameters
  that worked?

  Problem: this is a little too open-ended and un-motivating

