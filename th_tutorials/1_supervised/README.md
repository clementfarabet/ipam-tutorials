Supervised Learning
===================

In this tutorial, we're going to learn how to define a model, and train 
it using a supervised approach, to solve a multiclass classifaction task. Some of the material 
here is based on this [existing tutorial](http://torch.cogbits.com/doc/tutorials_supervised/).

The tutorial demonstrates how to:

  * pre-process the (train and test) data, to facilitate learning
  * describe a model to solve a classification task
  * choose a loss function to minimize
  * define a sampling procedure (stochastic, mini-batches), and apply one of several optimization techniques to train the model's parameters
  * estimate the model's performance on unseen (test) data

Each of these 5 steps is accompanied by a script, present in this directory:

  * 1_data.lua
  * 2_model.lua
  * 3_loss.lua
  * 4_train.lua
  * 5_test.lua

A top script, `doall.lua`, is also provided to run the complete procedure at once.

At the end of each section, I propose a couple of exercises, which are mostly
intended to make you modify the code, and get a good idea of the effect of each
parameter on the global procedure. Although the exercises are proposed at the end
of each section, they should be done after you've read the complete tutorial, as
they (almost) all require you to run the `doall.lua` script, to get training results.

The complete dataset is big, and we don't have time to play with the full set in
this short tutorial session. The script `doall.lua` comes with a `-size` flag, which
you should set to `small`, to only use 10,000 training samples.

The example scripts provided are quite verbose, on purpose. Instead of relying on opaque 
classes, dataset creation and the training loop are basically exposed right here. Although
a bit challenging at first, it should help new users quickly become independent, and able 
to tweak the code for their own problems.

On top of the scripts above, I provide an extra script, `A_slicing.lua`, which should help
you understand how tensor/arry slicing works in Torch (if you're a Matlab user,
you should be familiar with the contept, then it's just a matter of syntax).

Step 1: Data
------------

The code for this section is in `1_data.lua`. Run it like this:

```{.bash .numberLines}
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
    * Original images with character level bounding boxes.
    * MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).

We will be using the second format. In terms of dimensionality:

  * the inputs (images) are 3x32x32
  * the outputs (targets) are 10-dimensional

In this first section, we are going to preprocess the data to facilitate training.

The script provided automatically retrieves the dataset, all we have to do is load it:

```{.lua .numberLines}
-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return (#trainData.data)[1] end
}

loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return (#testData.data)[1] end
}
```

Preprocessing requires a floating point representation (the original
data is stored on bytes). Types can be easily converted in Torch, 
in general by doing: `dst = src:type('torch.TypeTensor')`, 
where `Type=='Float','Double','Byte','Int',`... Shortcuts are provided
for simplicity (`float(),double(),cuda()`,...):

```{.lua .numberLines}
trainData.data = trainData.data:float()
testData.data = testData.data:float()
```

We now preprocess the data. Preprocessing is crucial
when applying pretty much any kind of machine learning algorithm.

For natural images, we use several intuitive tricks:

  * images are mapped into YUV space, to separate luminance information from color information
  * the luminance channel (Y) is locally normalized, using a contrastive normalization operator: for each neighborhood, defined by a Gaussian kernel, the mean is suppressed, and the standard deviation is normalized to one.
  * color channels are normalized globally, across the entire dataset; as a result, each color component has 0-mean and 1-norm across the dataset.

```{.lua .numberLines}
-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(-std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(-std[i])
end

-- Local normalization
print '==> preprocessing data: normalize Y (luminance) channel locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(7)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood):float()

-- Normalize all Y channels locally:
for i = 1,trainData:size() do
   trainData.data[{ i,{1},{},{} }] = normalization(trainData.data[{ i,{1},{},{} }])
end
for i = 1,testData:size() do
   testData.data[{ i,{1},{},{} }] = normalization(testData.data[{ i,{1},{},{} }])
end
```

At this stage, it's good practice to verify that data is properly normalized:

```{.lua .numberLines}
for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end
```

We can then get an idea of how the preprocessing transformed the data by
displaying it:

```{.lua .numberLines}
-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

first256Samples_y = trainData.data[{ {1,256},1 }]
first256Samples_u = trainData.data[{ {1,256},2 }]
first256Samples_v = trainData.data[{ {1,256},3 }]
image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
image.display{image=first256Samples_u, nrow=16, legend='Some training examples: U channel'}
image.display{image=first256Samples_v, nrow=16, legend='Some training examples: V channel'}
```

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/1_supervised/img/y-channel.png)
![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/1_supervised/img/u-channel.png)
![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/1_supervised/img/v-channel.png)

### Exercise:

This is not the only kind of normalization! Data can be normalized in different manners, 
for instance, by normalizing individual features across the dataset (in this case, the
pixels). Try these different normalizations, and see the impact they have on the
training convergence.

Step 2: Model Definition
------------------------

The code for this section is in `2_model.lua`. Run it like this:

```{.bash .numberLines}
torch -i 2_model.lua -model linear
torch -i 2_model.lua -model mlp
torch -i 2_model.lua -model convnet
```

In this file, we describe three different models: convolutional neural networks (CNNs, or ConvNets), 
multi-layer neural networks (MLPs), and a simple linear model (which becomes a logistic
regression if used with a negative log-likelihood loss).

Linear regression is the simplest type of model. It is parametrized by a weight matrix W, 
and a bias vector b. Mathematically, it can be written as:

$$
y^n = Wx^n+b
$$

Using the _nn_ package, describing ConvNets, MLPs and other forms of sequential trainable models is really easy. All we have to do is create a top-level wrapper, which, as for the logistic regression, is going to be a sequential module, and then append modules into it. Implementing a simple linear model is therefore trivial:

```{.lua .numberLines}
model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add( nn.Linear(ninputs, noutputs) )
```

A slightly more complicated model is the multi-layer neural network (MLP). This model is
parametrized by two weight matrices, and two bias vectors:

$$
y^n = W_2 \text{sigmoid}(W_1 x^n + b_1) + b_2
$$

where the function _sigmoid_ is typically the symmetric hyperbolic tangent function. Again,
in Torch:

```{.lua .numberLines}
model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens,noutputs))
```

Compared to the linear regression model, the 2-layer neural network can learn arbitrary
non-linear mappings between its inputs and outputs. In practice, it can be quite hard
to train fully-connected MLPs to classify natural images.

Convolutional Networks are a particular form of MLP, which was tailored to efficiently learn to classify images. Convolutional Networks are trainable architectures composed of multiple stages. The input and output of each stage are sets of arrays called feature maps. For example, if the input is a color image, each feature map would be a 2D array containing a color channel of the input image (for an audio input each feature map would be a 1D array, and for a video or volumetric image, it would be a 3D array). At the output, each feature map represents a particular feature extracted at all locations on the input. Each stage is composed of three layers: a filter bank layer, a non-linearity layer, and a feature pooling layer. A typical ConvNet is composed of one, two or three such 3-layer stages, followed by a classification module. Each layer type is now described for the case of image recognition.

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/1_supervised/img/convnet.png)

Trainable hierarchical vision models, and more generally image processing algorithms are usually expressed as sequences of operations or transformations. They can be well described by a modular approach, in which each module processes an input image bank and produces a new bank. The figure above is a nice graphical illustration of this approach. Each module requires the previous bank to be fully (or at least partially) available before computing its output. This causality prevents simple parallelism to be implemented across modules. However parallelism can easily be introduced within a module, and at several levels, depending on the kind of underlying operations. These forms of parallelism are exploited in Torch7.

Typical ConvNets rely on a few basic modules:

  * Filter bank layer: the input is a 3D array with n1 2D feature maps of size n2 x n3. Each component is denoted $x_ijk$, and each feature map is denoted xi. The output is also a 3D array, y composed of m1 feature maps of size m2 x m3. A trainable filter (kernel) $k_ij$ in the filter bank has size l1 x l2 and connects input feature map x to output feature map $y_j$. The module computes $y_j = b_j + i_{kij} * x_i$ where $*$ is the 2D discrete convolution operator and $b_j$ is a trainable bias parameter. Each filter detects a particular feature at every location on the input. Hence spatially translating the input of a feature detection layer will translate the output but leave it otherwise unchanged.

  * Non-Linearity Layer: In traditional ConvNets this simply consists in a pointwise tanh() sigmoid function applied to each site (ijk). However, recent implementations have used more sophisticated non-linearities. A useful one for natural image recognition is the rectified sigmoid Rabs: $\abs(g_i.tanh())$ where $g_i$ is a trainable gain parameter. The rectified sigmoid is sometimes followed by a subtractive and divisive local normalization N, which enforces local competition between adjacent features in a feature map, and between features at the same spatial location.

  * Feature Pooling Layer: This layer treats each feature map separately. In its simplest instance, it computes the average values over a neighborhood in each feature map. Recent work has shown that more selective poolings, based on the LP-norm, tend to work best, with P=2, or P=inf (also known as max pooling). The neighborhoods are stepped by a stride larger than 1 (but smaller than or equal the pooling neighborhood). This results in a reduced-resolution output feature map which is robust to small variations in the location of features in the previous layer. The average operation is sometimes replaced by a max PM. Traditional ConvNets use a pointwise tanh() after the pooling layer, but more recent models do not. Some ConvNets dispense with the separate pooling layer entirely, but use strides larger than one in the filter bank layer to reduce the resolution. In some recent versions of ConvNets, the pooling also pools similar feature at the same location, in addition to the same feature at nearby locations.

Here is an example of ConvNet that we will use in this tutorial:

```{.lua .numberLines}
-- parameters
nstates = {16,256,128}
fanin = {1,4}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

-- Container:
model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMap(nn.tables.random(nfeats, nstates[1], fanin[1]), filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(16, normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMap(nn.tables.random(nstates[1], nstates[2], fanin[2]), filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[3], noutputs))
```

A couple of comments about this model:

  * the input has 3 feature maps, each 32x32 pixels. It is the convention for all nn.Spatial* layers to work on 3D arrays, with the first dimension indexing different features (here normalized YUV), and the next two dimensions indexing the height and width of the image/map.

  * the fist layer applies 16 filters to the input map, each being 5x5. The receptive field of this first layer is 5x5, and the maps produced by it are therefore 16x28x28. This linear transform is then followed by a non-linearity (tanh), and an L2-pooling function, which pools regions of size 2x2, and uses a stride of 2x2. The result of that operation is a 16x14x14 array, which represents a 14x14 map of 16-dimensional feature vectors. The receptive field of each unit at this stage is 7x7.

  * the second layer is very much analogous to the first, except that now the 16-dim feature maps are projected into 256-dim maps, with a fully-connected connection table: each unit in the output array is influenced by a 4x5x5 neighborhood of features in the previous layer. That layer has therefore 4x256x5x5 trainable kernel weights (and 256 biases). The result of the complete layer (conv+pooling) is a 256x5x5 array.

  * at this stage, the 5x5 array of 256-dimensional feature vectors is flattened into a 6400-dimensional vector, which we feed to a two-layer neural net. The final prediction (10-dimensional distribution over classes) is influenced by a 32x32 neighborhood of input variables (YUV pixels).

  * recent work (Jarret et al.) has demonstrated the advantage of locally normalizing sets of internal features, at each stage of the model. The use of smoother pooling functions, such as the L2 norm for instance instead of the harsher max-pooling, has also been shown to yield better generalization (Sermanet et al.). We use these two ingredients in this model.

  * one other remark: it is typically not a good idea to use fully connected layers, in internal layers. In general, favoring large numbers of features (over-completeness) over density of connections helps achieve better results (empirical evidence of this was reported in several papers, as in Hadsell et al.). The SpatialConvolutionMap module accepts tables of connectivities (maps) that allows one to create arbitrarily sparse connections between two layers. A couple of standard maps/tables are provided in nn.tables.

### Exercises:

The number of meta-parameters to adjust can be daunting at first. Try to get a feeling
of the inlfuence of these parameters on the learning convergence:

  * going from the MLP to a ConvNet of similar size (you will need to think a little bit about the equivalence between the ConvNet states and the MLP states)

  * replacing the 2-layer MLP on top of the ConvNet by a simpler linear classifier

  * replacing the L2-pooling function by a max-pooling

  * replacing the two-layer ConvNet by a single layer ConvNet with a much larger pooling area (to conserve the size of the receptive field)

Step 3: Loss Function
---------------------

Now that we have a model, we need to define a loss function to be minimized, across the entire training set:

$$
L = \sum_n l(y^n,t^n)
$$

One of the simplest loss functions we can minimize is the mean-square error between the predictions (outputs of the model), and the groundtruth labels, across the entire dataset:

$$
l(y^n,t^n) = \frac{1}{2} \sum_i (y_i^n - t_i^n)^2
$$

or, in Torch:

```{.lua .numberLines}
criterion = nn.MSECriterion()
```

The MSE loss is typically not a good one for classification, as it forces the model to exactly predict the values imposed by the targets (labels). 

Instead, a more commonly used, probabilistic objective is the negative log-likelihood. To minimize a negative log-likelihood, we first need to turn the predictions of our models into properly normalized log-probabilities. For the linear model, this is achieved by feeding the output units into a _softmax_ function, which turns the linear regression into a logistic regression:

$$
P(Y=i|x^n,W,b) = \text{softmax}(Wx^n+be) \\
\\
P(Y=i|x^n,W,b) = \frac{ e^{Wx_i^n+b} }{ \sum_j e^{Wx_j^n+b} }
$$

As we're interested in classification, the final prediction is then achieved by taking the argmax of this distribution:

$$
y^n = \arg\max_i P(Y=i|x^n,W,b)
$$

in which case the ouput y is a scalar.

More generally, the output of any model can be turned into normalized log-probabilities, by stacking
a _softmax_ function on top. So given any of the models defined above, we can simply do:

```{.lua .numberLines}
model:add( nn.LogSoftMax() )
```

We want to maximize the likelihood of the correct (target) class, for each sample in the dataset. This is equivalent to minimizing the negative log-likelihood (NLL), or minimizing the cross-entropy between the predictions of our model and the targets (training data). Mathematically, the per-sample loss can be defined as:

$$
l(x^n,t^n) = -\log(P(Y=t^n|x^n,W,b))
$$

Given that our model already produces log-probabilities (thanks to the _softmax_), the loss is quite straightforward to estimate. In Torch, we use the _ClassNLLCriterion_, which expects its input as being a vector of log-probabilities, and the target as being an integer pointing to the correct class:

```{.lua .numberLines}
criterion = nn.ClassNLLCriterion()
```

Finally, another type of classification loss is the multi-class margin loss, which is closer to the well-known SVM loss. This loss function doesn't require normalized outputs, and can be implemented like this:

```{.lua .numberLines}
criterion = nn.MultiMarginCriterion()
```

The margin loss typically works on par with the negative log-likelihood. I haven't tested this thoroughly, so it's time for more exercises.

### Exercises:

The obvious exercise now is to play with these different loss functions, and see how they affect convergence. In particular try to:

  * swap the loss from NLL to MultiMargin, and if it doesn't work as well, thinkg a little bit more about the scaling of the gradients, and whether you should rescale the learning rate.

Step 4: Training Procedure
--------------------------

We now have some training data, a model to train, and a loss function to minimize. We define a training procedure, which you will find in this file: `4_train.lua`.

A very important aspect about supervised training of non-linear models (ConvNets and MLPs) is the fact that the optimization problem is not convex anymore. This reinforces the need for a stochastic estimation of gradients, which have shown to produce much better generalization results for several problems.

In this example, we show how the optimization algorithm can be easily set to either L-BFGS, CG, SGD or ASGD. In practice, it's very important to start with a few epochs of pure SGD, before switching to L-BFGS or ASGD (if switching at all). The intuition for that is related to the non-convex nature of the problem: at the very beginning of training (random initialization), the landscape might be highly non-convex, and no assumption should be made about the shape of the energy function. Often, SGD is the best we can do. Later on, batch methods (L-BFGS, CG) can be used more safely.

Interestingly, in the case of large convex problems, stochasticity is also very important, as it allows much faster (rough) convergence. Several works have explored these techniques, in particular, this recent [paper from Byrd/Nocedal](http://users.eecs.northwestern.edu/~nocedal/PDFfiles/dss.pdf), and work on pure stochastic gradient descent by [Bottou](http://leon.bottou.org/projects/sgd).

Here is our full training function, which demonstrates that you can switch the optimization you're using at runtime (if you want to), and also modify the batch size you're using at run time. You can do all these things because we create the evaluation closure each time we create a new batch. If the batch size is 1, then the method is purely stochastic. If the batch size is set to the complete dataset, then the method is a pure batch method.

```{.lua .numberLines}
-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

-- Training function
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]:double()
         local target = trainData.labels[shuffle[i]]
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = trsize * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   confusion:zero()

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end
```

We could then run the training procedure like this:

```{.lua .numberLines}
while true
   train()
end
```

### Exercices:

So, a bit on purpose, I've given you this blob of training code with rather few
explanations. Try to understand what's going on, to do the following things:

  * modify the batch size (and possibly the learning rate) and observe the impact
  on training accuracy, and test accuracy (generalization)

  * change the optimization method, and in particular, try to start with L-BFGS
  from the very first epoch. What happens then?

Step 5: Test the Model
----------------------

A common thing to do is to test the model's performance while we train it. Usually, this
test is done on a subset of the training data, that is kept for validation. Here
we simply define the test procedure on the available test set:

```{.lua .numberLines}
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]:double()
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   confusion:zero()

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end
```

The train/test procedure now looks like this:

```{.lua .numberLines}
while true
   train()
   test()
end
```

### Exercices:

As mentionned above, validation is the proper (an only!) way to train
a model and estimate how well it does on unseen data:

  * modify the code above to extract a subset of the training data to 
  use for validation

  * once you have that, add a stopping condition to the script, such
  that it terminates once the validation error starts rising above
  a certain threshold. This is called early-stopping.

All Done!
---------

The final step of course, is to run _doall.lua_, which will train the model
over the entire training set. By default, it uses the basic training set size
(about 70,000 samples). If you use the flag: `-size extra`, you will obtain
state-of-the-art results (in a couple of days of course!).

### Final Exercise

If time allows, you can try to replace this dataset by other datasets, such
as MNIST, which you should already have working (from day 1). Try to think
about what you have to change/adapt to work with other types of images
(non RGB, binary, infrared?).

Tips, going futher
------------------

### Tips and tricks for MLP training

There are several hyper-parameters in the above code, which are not (and,
generally speaking, cannot be) optimized by gradient descent.
The design of outer-loop algorithms for optimizing them is a topic of ongoing
research.
Over the last 25 years, researchers have devised various rules of thumb for choosing them.
A very good overview of these tricks can be found in [Efficient
BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCun,
Leon Bottou, Genevieve Orr, and Klaus-Robert Mueller. Here, we summarize
the same issues, with an emphasis on the parameters and techniques that we
actually used in our code.

### Tips and Tricks: Nonlinearity

Which non-linear activation function should you use in a neural network?
Two of the most common ones are the logistic sigmoid and the tanh functions.
For reasons explained in [Section 4.4](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), nonlinearities that
are symmetric around the origin are preferred because they tend to produce
zero-mean inputs to the next layer (which is a desirable property).
Empirically, we have observed that the tanh has better convergence properties.

### Tips and Tricks: Weight initialization

At initialization we want the weights to be small enough around the origin
so that the activation function operates near its linear regime, where gradients are
the largest. Otherwise, the gradient signal used for learning is attenuated by
each layer as it is propagated from the classifier towards the inputs.
Proper weight initialization is implemented in all the modules provided in `nn`, 
so you don't have to worry about it. Each module has a `reset()` method,
which initializes the parameter with a uniform distribution that takes 
into account the fanin/fanout of the module. It's called by default when
you create a new module, but you can call it at any time to reset the weights.

### Tips and Tricks: Learning Rate

Optimization by stochastic gradient descent is very sensitive to the step 
size or _learning rate_. There is a great deal of literature on how to choose 
a the learning rate, and how to change it during optimization. A good
heuristic is to use a `lr_0/(1+t*decay)` decay on the learning, where you
set the decay to a value that's inversely proportional to the number of 
samples you want to see with an almost flat learning rate, before starting
decaying exponentially.

[Section 4.7](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) details
procedures for choosing a learning rate for each parameter (weight) in our
network and for choosing them adaptively based on the error of the classifier.

### Tips and Tricks: Number of hidden units

The number of hidden units that gives best results is dataset-dependent.
Generally speaking, the more complicated the input distribution is, the more
capacity the network will require to model it, and so the larger the number of 
hidden units that will be needed.

### Tips and Tricks: Norm Regularization

Typical values to try for the L1/L2 regularization parameter are 10^-2 or 10^-3.
It is usually only useful to regularize the topmost layers of the MLP (closest
to the classifier), if not the classifier only. An L2 regularization is really
easy to implement, `optim.sgd` provides an implementation, but it's global
to the parameters, which is typically not a good idea. Instead, after each call
to `optim.sgd`, you can simply apply the regularization on the subset of weights 
of interest:

```{.lua .numberLines}
-- model:
model = nn.Sequential()
model:add( nn.Linear(100,200) )
model:add( nn.Tanh() )
model:add( nn.Linear(200,10) )

-- weights to regularize:
reg = {}
reg[1] = model:get(3).weight
reg[2] = model:get(3).bias

-- optimization:
while true do
   -- ...
   optim.sgd(...)

   -- after each optimization step (gradient descent), regularize weights
   for _,w in ipairs(reg) do
      w:add(-weightDecay, w)
   end
end
```

### Tips and tricks for ConvNet training

ConvNets are especially tricky to train, as they add even more hyper-parameters than
a standard MLP. While the usual rules of thumb for learning rates and regularization 
constants still apply, the following should be kept in mind when optimizing ConvNets.

#### Number of filters

Since feature map size decreases with depth, layers near the input layer will 
tend to have fewer filters while layers higher up can have much more. In fact, to
equalize computation at each layer, the product of the number of features
and the number of pixel positions is typically picked to be roughly constant
across layers. To preserve the information about the input would require
keeping the total number of activations (number of feature maps times
number of pixel positions) to be non-decreasing from one layer to the next
(of course we could hope to get away with less when we are doing supervised
learning). The number of feature maps directly controls capacity and so
that depends on the number of available examples and the complexity of 
the task.

#### Filter Shape

Common filter shapes found in the literature vary greatly, usually based on
the dataset. Best results on MNIST-sized images (28x28) are usually in the 
5x5 range on the first layer, while natural image datasets (often with hundreds 
of pixels in each dimension) tend to use larger first-layer filters of shape 
7x7 to 12x12.

The trick is thus to find the right level of "granularity" (i.e. filter
shapes) in order to create abstractions at the proper scale, given a
particular dataset.

It's also possible to use multiscale receptive fields, to allow the ConvNet
to have a much larger receptive field, yet keeping its computational complexity
low. This type of procedure was proposed for scene parsing (where context
is crucial to recognize objects) in 
[this paper](http://data.clement.farabet.net/pubs/icml12.pdf).

#### Pooling Shape

Typical values for pooling are 2x2. Very large input images may warrant
4x4 pooling in the lower-layers. Keep in mind however, that this will reduce the
dimension of the signal by a factor of 16, and may result in throwing away too
much information. In general, the pooling region is independent from the stride
at which you discard information. In Torch, all the pooling modules (L2, average, 
max) have separate parameters for the pooling size and the strides, for
example:

```{.lua .numberLines}
nn.SpatialMaxPooling(pool_x, pool_y, stride_x, stride_y)
```
