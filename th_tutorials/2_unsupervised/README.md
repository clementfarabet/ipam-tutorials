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

As for the supervised learning tutorial, the code is split into multiple files:

  * 1_data.lua
  * 2_models.lua
  * 3_train.lua

And a top file, `doall.lua`, runs the complete experiment. In this case though, there are
too many inter-dependencies between files, so they can be loaded individually. If you still want
to interact with the code, run:

```lua
torch -i doall.lua
```

And issue ctrl+C at anytime, to kill the execution. You will then have access to an interpreter,
and be able to explore any of the variables (model, data, ...).

The complete tutorial uses a version of the Berkeley dataset, which is already locally normalized
(preprocessed), with 56x56 patches randomly extracted from the original images.

Step 1: Models & Loss functions
-------------------------------

### Basic Autoencoders

An autoencoder is a model that takes a vector input y, maps it into a hidden representation z (code) using an encoder which typically has this form:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/auto_encoder.png)

where s is a non-linear activation function (the tanh function is a common choice), W_e the encoding matrix and b_e a vector of bias parameters.

The hidden representation z, often called code, is then mapped back into the space of y, using a decoder of this form:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/auto_decoder.png)

where W_d is the decoding matrix and b_d a vector of bias parameters.

The goal of the autoencoder is to minimize the reconstruction error, which is represented by a distance between y and y~. The most common type of distance is the mean squared error:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/mse_loss.png)

The code z typically has less dimensions than y, which forces the autoencoder to learn a good representation of the data. In its simplest form (linear), an autoencoder learns to project the data onto its first principal components. If the code z has as many components as y, then no compression is required, and the model could typically end up learning the identity function. Now if the encoder has a non-linear form (using a tanh, or using a multi-layered model), then the autoencoder can learn a potentially more powerful representation of the data.

To describe the model, we use the _unsup_ package, which provides templates to build autoencoders.

The first step is to describe an encoder, which we can do by using any of the modules available in nn:

```lua
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,outputSize))
encoder:add(nn.Tanh())
```

The second step is to describe the decoder, a simple linear module:

```lua
decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize,inputSize))
```

Finally, we use the built-in AutoEncoder class from unsup, which automatically provides a mean-square error loss:

```lua
module = unsup.AutoEncoder(encoder, decoder, params.beta)
```

At this stage, estimating the loss (reconstruction error) can be done like this, for arbitrary inputs:

```lua
input = torch.randn(inputSize)
loss = module:updateOutput(input,input)
```

Note that we need to provide the input, and a target that we wish to reconstruct. In this case the target is the input, but in some cases, we might want to provide a noisy version of the input, and force the autoencoder to predict the correct input (this is what denoising autoencoders do).

As for any nn module, gradients can be estimated this way:

```lua
-- get parameters and gradient pointers
x,dl_dx = module:getParameters()
 
-- compute loss
loss = module:updateOutput(inputs[i], targets[i])
 
-- compute gradients wrt input and weights
dl_dx:zero()
module:updateGradInput(input, input)
module:accGradParameters(input, input)
 
-- at this stage, dl_dx contains the gradients of the loss wrt
-- the trainable parameters x
```

One serious potential issue with auto-encoders is that if there is no other
constraint besides minimizing the reconstruction error,
then an auto-encoder with n inputs and an encoding of dimension at least n could potentially 
just learn the identity function, and fail to differentiate
test examples (from the training distribution) from other input configurations.

Surprisingly, experiments reported in `Bengio 2007` nonetheless
suggest that in practice, when trained with stochastic gradient descent, 
non-linear auto-encoders with more hidden units
than inputs (called overcomplete) yield useful representations
(in the sense of classification error measured on a network taking this
representation in input). A simple explanation is based on the
observation that stochastic gradient
descent with early stopping is similar to an L2 regularization of the
parameters. To achieve perfect reconstruction of continuous
inputs, a one-hidden layer auto-encoder with non-linear hidden units
(exactly like in the above code)
needs very small weights in the first (encoding) layer (to bring the non-linearity of
the hidden units in their linear regime) and very large weights in the
second (decoding) layer.

With binary inputs, very large weights are also needed to completely minimize the 
reconstruction error. Since the implicit or explicit regularization makes it difficult 
to reach large-weight solutions, the optimization algorithm finds encodings which
only work well for examples similar to those in the training set, which is
what we want. It means that the representation is exploiting statistical
regularities present in the training set, rather than learning to
replicate the identity function.

#### Exercises:

There are 3 main parameters you can play with: the input size, the output
size, and the type of non-linearity you use.

  * observe the effect of the input size (past a certain size, it's almost
  impossible to reconstruct anything, unless you have as many output units
  as inputs)

  * observe the effect of the output size, and why not compare the results
  with PCA?

  * finally, as noted above, regularization is very important for simple
  autoencoders. Can you implement an L2 regularization? We saw an example
  of regularizaiton in the previous tutorial on supervised training.

The autoencoder code I provide here is very simple and naive. A natural
extension of autoencoders are denoising autoencoders, where the idea is
to add salt and pepper noise to the input variables, to force the mode
to construct a good representation for the data. Can you implement this?


### Predictive Sparse Decomposition (PSD) Autoencoder

One big shortcoming of basic autoencoders is that it's usually hard to train them, and hard to avoid getting to close to learning the identity function. In practice, using a code y that is smaller than x is enough to avoid learning the identity, but it remains hard to do much better than PCA.

Using codes that are overcomplete (i.e. with more components than the input) makes the problem even worse. There are different ways that an autoencoder with an overcomplete code may still discover interesting representations. One common way is the addition of sparsity: by forcing units of the hidden representation to be mostly 0s, the autoencoder has to learn a nice distributed representation of the data.

We now present a method to impose sparsity on the code, which typically allows codes that are overcomplete, sparse, and very useful for tasks like classification/recognition.

Adaptive sparse coding methods learn a possibly overcomplete set of basis functions, such that natural image patches can be reconstructed by linearly combining a small subset of these bases. The applicability of these methods to visual object recognition tasks has been limited because of the prohibitive cost of the optimization algorithms required to compute the sparse representation.

In this tutorial we propose a simple and efficient algorithm to learn overcomplete basis functions, by introducing a particular form of autoencoder. After training, the model also provides a fast and smooth approximator to the optimal representation, achieving even better accuracy than exact sparse coding algorithms on visual object recognition tasks.

#### Sparse Coding

Finding a representation z in R^m for a given signal y in R^n by linear combination of an overcomplete set of basis vectors, columns of matrix B with m > n, has infinitely many solutions. In optimal sparse coding, the problem is formulated as:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/sparse_coding.png)

where the l0 norm is defined as the number of non-zero elements in a given vector. This is a combinatorial problem, and a common approximation of it is the following optimization problem:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/sparse_coding_optim.png)

This particular formulation, called Basis Pursuit Denoising, can be seen as minimizing an objective that penalizes the reconstruction error using a linear basis set and the sparsity of the corresponding representation. While this formulation is nice, inference requires running some sort of iterative minimization algorithm that is always computationally expensive. In the following we present a predictive version of this algorithm, based on an autoencoder formulation, which yields fixed-time, and fast inference.

#### Linear PSD

In order to make inference efficient, we train a non-linear regressor that maps input signals y to sparse representations z. We consider the following nonlinear mapping:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/psd_encoder.png)

where W is a weight trainable matrix, d a trainable vector of biases, and g a vector of gains. We want to train this nonlinear mapping as a predictor for the optimal solution to the sparse coding algorithm presented in the previsous section.

The following loss function, called predictive sparse decomposition, can help us achieve such a goal:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/psd_loss.png)

The first two terms are the basic sparse coding presented above, while the 3rd term is our predictive sparse regressor. Minimizing this loss yields an encoder that produces sparse decompositions of the input signal.

With the unsup package, this can be implemented very simply.

We define an encoder first:

```lua
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,outputSize))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputSize))
```

Then the decoder is the L1 solution presented above:

```lua
decoder = unsup.LinearFistaL1(inputSize, outputSize, params.lambda)
```

Under the hood, this decoder relies on FISTA to find the optimal sparse code. FISTA is available in the optim package.

Finally, both modules can be packaged together into an autoencoder. We can't use the basic AutoEncoder class to do this, because the LinearFistaL1 decoder is a bit peculiar. Insted, we use a special-purpose PSD container:

```lua
module = unsup.PSD(encoder, decoder)
```

#### Convolutional PSD

For vision/image applications, fully connected linear autoencoders are often overkill, in their number of trainable parameters. Using convolutional filters, inspired by convolutional networks (see supervised learning tutorial on ConvNets) can help learn much better filters for vision.

A convolutional version of the PSD autoencoder can be derived by simply replacing the encoder and decoder by convolutional counterparts:

```lua
-- connection table:
conntable = nn.tables.full(1, 32)
 
-- decoder's table:
local decodertable = conntable:clone()
decodertable[{ {},1 }] = conntable[{ {},2 }]
decodertable[{ {},2 }] = conntable[{ {},1 }]
local outputSize = conntable[{ {},2 }]:max()
 
-- encoder:
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolutionMap(conntable, 5, 5))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputSize))
 
-- decoder is L1 solution:
decoder = unsup.SpatialConvFistaL1(decodertable, 5, 5, 25, 25)
```

#### Exercises:

As for the regular autoencoders, you can play with the input size, but what mostly affects
the models' performance now is the filter size (in its convolutional version), and
the sparsity coefficient. Try to see how the sparsity coefficient affects the 
(encoder) weights. Do you observe a checkerboard effect? Why?

Step 2: Training
----------------

If you've read the tutorial on supervised learning, training a model unsupervised is basically equivalent. We first define a closure that computes the loss, and the gradients of that loss wrt the trainable parameters, and then pass this closure to one of the optimizers in optim. As usual, we use SGD to train autoencoders on large amounts of data:

```lua
-- some parameters
local minibatchsize = 50
 
-- parameters
x,dl_dx = module:getParameters()
 
-- SGD config
sgdconf = {learningRate = 1e-3}
 
-- assuming a table trainData with the form:
-- trainData = {
--    [1] = sample1,
--    [2] = sample2,
--    [3] ...
-- }
for i = 1,#trainData,minibatchsize do
 
    -- create minibatch of training samples
    samples = torch.Tensor(minibatchsize,inputSize)
    for i = 1,minibatchsize do
        samples[i] = trainData[i]
    end
 
    -- define closure
    local feval = function()
      -- reset gradient/f
      local f = 0
      dl_dx:zero()
 
      -- estimate f and gradients, for minibatch
      for i = 1,minibatchsize do
         -- f
         f = f + module:updateOutput(samples[i], samples[i])
 
         -- gradients
         module:updateGradInput(samples[i], samples[i])
         module:accGradParameters(samples[i], samples[i])
      end
 
      -- normalize
      dl_dx:div(minibatchsize)
      f = f/minibatchsize
 
      -- return f and df/dx
      return f,dl_dx
   end
 
   -- do SGD step
   optim.sgd(feval, x, sgdconf)
 
end
```

Ok, that's it, given some training data, this code will loop over all samples, and minimize the reconstruction error, using stochastic gradient descent.

#### Exercises:

As usual, we have access to several optimization techniques, and can set the batch size. How does the batch size affect the reconstruction error?


Step 3: Second-order Information
--------------------------------

Training PSD autoencoders, especially convolutional ones, is a particularly challenging 
optimization problem. This is due to the loss function, which contains particularly opposing
terms. As a result, the encoder's weights usually get stuck with checker board patterns
for a long time, before starting to properly minimize the cost function. To east the optimization
a little bit, we can use second-order information, to condition each parameter individually.

A simple way of doing that, which was proposed in (Lecun, 1987), is to compute an approximation
of the diagonal terms of the Hessian, and adapt the learning rate of each parameter in the system
proportionally to the inverse of the diagonal terms.

Computing the diagonal terms is very analogous to computing the gradients themselves:

```lua
if params.hessian and math.fmod(t , params.hessianinterval) == 1 then
  -- some extra vars:
  local hessiansamples = params.hessiansamples
  local minhessian = params.minhessian
  local maxhessian = params.maxhessian
  local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
  etas = etas or ddl_ddx:clone()

  print('==> estimating diagonal hessian elements')
  for i = 1,hessiansamples do
     -- next
     local ex = dataset[i]
     local input = ex[1]
     local target = ex[2]
     module:updateOutput(input, target)

     -- gradient
     dl_dx:zero()
     module:updateGradInput(input, target)
     module:accGradParameters(input, target)

     -- hessian
     ddl_ddx:zero()
     module:updateDiagHessianInput(input, target)
     module:accDiagHessianParameters(input, target)

     -- accumulate
     ddl_ddx_avg:add(1/hessiansamples, ddl_ddx)
  end

  -- cap hessian params
  print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
  ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
  ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
  print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())

  -- generate learning rates
  etas:fill(1):cdiv(ddl_ddx_avg)
end
```

The result of this code is a vector _etas_, which contains a list of learning rates, one
for each trainable parameter in the system. These _etas_ can be re-estimated every
once in a while, depending on the complexity of the model / data. They can then
simply be passed to the SGD function, which will use them as individual learning rates:

```lua
sgdconf = sgdconf or {learningRate = params.eta,
                      learningRateDecay = params.etadecay,
                      learningRates = etas,
                      momentum = params.momentum}
_,fs = optim.sgd(feval, x, sgdconf)
```

Some Results
------------

Here's a snapshot that shows PSD learning 256 encoder filters, after seeing 
80,000 training patches (9x9), randomly sampled from the Berkeley dataset.

Initial weights:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/filters_00000.png)

At 40,000 samples:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/filters_40000.png)

At 80,000 samples:

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/2_unsupervised/img/filters_80000.png)

Notice how the checker board effect is still present after 40,000 samples, but completely
gone at the end.

#### Exercises:

Given the small amount of time, try to reproduce these results, but with only
a handful of filters (say 8 or 16).
