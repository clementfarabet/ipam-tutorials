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

