Supervised Learning
===================

In this tutorial, we're going to learn how to define a model, and train 
it using a supervised approach, to solve a multiclass classifaction task. Some of the material 
here is based on this [existing tutorial](http://torch.cogbits.com/doc/tutorials_supervised/).

The tutorial demonstrates how to:
  * pre-process the (train and test) data, to facilitate learning
  * describe a model to solve a classificaiton task
  * choose a loss function to minimize
  * define a sampling procedure (stochastic, mini-batches), and apply one of several optimization techniques to train the model's parameters
  * estimate the model's performance on unseen (test) data

Each of these 5 steps is accompanied by a script, present in this directory:

  * 1_data.lua
  * 2_model.lua
  * 3_loss.lua
  * 4_train.lua
  * 5_test.lua

A top script, doall.lua, is also provided to run the complete procedure at once.

The example scripts provided are quite verbose, on purpose. Instead of relying on opaque 
classes, dataset creation and the training loop are basically exposed right here. Although
a bit challenging at first, it should help new users quickly become independent, and able 
to tweak the code for their own problems.

On top of the scripts above, I provide an extra script, A_slicing.lua, which should help
you understand how tensor/arry slicing works in Torch.

Step 1: Data
------------

The code for this section is in *1_data.lua*. Run it like this:

```bash
torch -i 1_data.lua
```

This will give you an interpreter to play with the data once it's loaded/preprocessed.

For this tutorial, we'll be using the [Street View House Number](http://ufldl.stanford.edu/housenumbers/)
dataset. SVHN is a real-world image dataset for developing machine learning and object recognition 
algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar 
in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of 
magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, 
unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is 
obtained from house numbers in Google Street View images.

Overview of the dataset:

  * 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
  * 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data
  * Comes in two formats:
    * 1. Original images with character level bounding boxes.
    * 2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).

We will be using the second format. In terms of dimensionality:

  * the inputs (images) are 3x32x32
  * the outputs (targets) are 10-dimensional

In this first section, we are going to preprocess the data to facilitate the training.

Step 2: Model Definition
------------------------

We introduce several classical models: convolutional neural networks (CNNs, or ConvNets), 
multi-layer neural networks (MLPs), and simple logistic regression units.

Step 3: Loss Function
---------------------

Step 4: Training Procedure
--------------------------

Step 5: Test the Model
----------------------

