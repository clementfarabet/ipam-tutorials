Unsupervised Learning
=====================

In this tutorial, we're going to learn how to define a model, and train 
it using an unsupervised approach, with the goal of learning good features/representation
for classification tasks. Some of the material 
here is based on this [existing tutorial](http://torch.cogbits.com/doc/tutorials_unsupervised/).

The tutorial demonstrates how to:
  * describe an unsupervised model
  * define a (multi-term) loss function to minimize
  * define a sampling procedure (stochastic, mini-batches), and apply one of several optimization techniques to train the model's parameters
  * use second-order information (diagonal of the hessian) to ease the optimization procedure

Step 1: Models & Loss functions
-------------------------------

Step 2: Training
----------------

Step 3: Second-order Information
--------------------------------

Some Results
------------

Here's a snapshot that shows PSD learning 256 encoder filters, after seeing 
80,000 training patches (9x9), randomly sampled from the Berkeley dataset.

Initial weights:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/filters_00000.jpg)

At 40,000 samples:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/filters_40000.jpg)

At 80,000 samples:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/filters_80000.jpg)

