Going Further
=============

In the previous 3 tutorials, you saw 3 complete applications of machine
learning: supervised learning of MLPs and ConvNets, unsupervised pre-training
of auto-encoders, and supervised training of conditionnal random fields.

In this tutorial, we will see how to extend what we've seen, mainly by
creating our own new modules, and testing them, and then see how we can
use the GPU to speed up learning, and testing.

Defining your own Neural Net Module
-----------------------------------

So far we've used existing modules. In this section, we'll define two
new modules, and see how simple it is to do so.

Modules are bricks to build neural networks. A `Module` is a neural network by itself, 
but it can be combined with other networks using container classes to create complex
neural networks. `Module` is an abstract class which defines fundamental methods necessary 
for a training a neural network. All modules are serializable.

Modules contain two states variables: `output` and `gradInput`. Here we review
the set of basic functions that a `Module` has to implement:

#### `[output] forward(input)`

Takes an input object, and computes the corresponding output of the module. In general 
input and output are `Tensors`. However, some special sub-classes like table layers might 
expect something else. Please, refer to each module specification for further information.

After a `forward()`, the output state variable should have been updated to the new value.

It is not advised to override this function. Instead, one should implement updateOutput(input) 
function. The forward module in the abstract parent class `Module` will call updateOutput(input).

#### `[gradInput] backward(input, gradOutput)`

Performs a backpropagation step through the module, with respect to the given input. In 
general this method makes the assumption forward(input) has been called before, with the same 
input. This is necessary for optimization reasons. If you do not respect this rule, `backward()`
will compute incorrect gradients.

In general input and gradOutput and gradInput are `Tensors`. However, some special sub-classes 
like table layers might expect something else. Please, refer to each module specification 
for further information.

A backpropagation step consist in computing two kind of gradients at input given gradOutput 
(gradients with respect to the output of the module). This function simply performs this 
task using two function calls:

  * A function call to updateGradInput(input, gradOutput).
  * A function call to accGradParameters(input,gradOutput).

It is not advised to override this function call in custom classes. It is better to 
override `updateGradInput(input, gradOutput)` and `accGradParameters(input, gradOutput)`
functions.

#### `[output] updateOutput(input, gradOutput)`

When defining a new module, this method should be overloaded.

Computes the output using the current parameter set of the class and input. This function returns the result which is stored in the output field.

#### `[gradInput] updateGradInput(input, gradOutput)`

When defining a new module, this method should be overloaded.

Computing the gradient of the module with respect to its own input. This is returned in gradInput. 
Also, the gradInput state variable is updated accordingly.

#### `[gradInput] accGradParameters(input, gradOutput)`

When defining a new module, this method may need to be overloaded, if the module has
trainable parameters.

Computing the gradient of the module with respect to its own parameters. Many modules do 
not perform this step as they do not have any parameters. The state variable name for the 
parameters is module dependent. The module is expected to accumulate the gradients with 
respect to the parameters in some variable.

Zeroing this accumulation is achieved with zeroGradParameters() and updating the parameters 
according to this accumulation is done with updateParameters().

#### `reset()`

This method defines how the trainable parameters are reset, _i.e._ initialized before
training.

---------------------------------------------------------------------------------------------

Modules provide a few other methods that you might want to define, if you are not planing
to use the `optim` package. These methods help `zero()` the parameters, and update them
using very basic techniques.

In terms of code structure, `Torch` provides a class model, which we use for inheritance,
and in general for the definition of all the modules in `nn`. Here is an empty holder
for a typical new class:

```{.lua .numberLines}
local NewClass, Parent = torch.class('nn.NewClass', 'nn.Module')

function NewClass:__init()
   parent.__init(self)
end

function NewClass:updateOutput(input)
end

function NewClass:updateGradInput(input, gradOutput)
end

function NewClass:accGradParameters(input, gradOutput)
end

function NewClass:reset()
end
```

When defining a new class, all we need to do is fill in these empty functions. Note
that when defining the constructor `__init()`, we always call the parent's
constructor first.

Let's see some practical examples now.

### Dropout Activation Units

This week we heard about dropout activation units. The idea there is to perturbate
the activations of hidden units, by randomly zeroing some of these units. 

Such a class could be defined like this:

```{.lua .numberLines}
local Dropout, Parent = torch.class('nn.Dropout', 'nn.Module')

function Dropout:__init(percentage)
   Parent.__init(self)
   self.p = percentage or 0.5
   if self.p > 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p <= 1')
   end
end

function Dropout:updateOutput(input)
   self.noise = torch.rand(input:size()) -- uniform noise between 0 and 1
   self.noise:add(1 - self.p):floor()  -- a percentage of noise
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   return self.gradInput
end
```

The file is provided in this directory, in `Dropout.lua`. The script
`1_dropout.lua` demonstrates how to create an instance of this module, 
and test it on some data (lena):

```{.lua .numberLines}
-- in this file, we test the dropout module we've defined:
require 'nn'
require 'Dropout'
require 'image'

-- define a dropout object:
n = nn.Dropout(0.5)

-- load an image:
i = image.lena()

-- process the image:
result = n:forward(i)

-- display results:
image.display{image=i, legend='original image'}
image.display{image=result, legend='dropout-processed image'}

-- some stats:
mse = i:dist(result)
print('mse between original imgae and dropout-processed image: ' .. mse)
```

When writing modules with gradient estimation, it's always very important
to test your implementation. This can be easily done using the `Jacobian`
class provided in `nn`, which compares the implementation of the gradient
methods (`updateGradInput()` and `accGradParameters()`) with the Jacobian
matrix obtained by finite differences (perturbating the input of the module,
and estimating the deltas on the output). This can be done like this:

```{.lua .numberLines}
-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 0.5
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.Dropout(percentage)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
```

One slight issue with the `Jacobian` class is the fact that it assumes
that the outputs of a module are deterministic wrt to the inputs. This is
not the case for that particular module, so for the purpose of these tests
we need to freeze the noise generation, _i.e._ do it only once:

```{.lua .numberLines}
-- we overload the updateOutput() function to generate noise only
-- once for the whole test.
function nn.Dropout.updateOutput(self, input)
   self.noise = self.noise or torch.rand(input:size()) -- uniform noise between 0 and 1
   self.noise:add(1 - self.p):floor()  -- a percentage of noise
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   return self.output
end
```

### Exercise

Well, at this stage, a natural exercise would be to try to integrate this
module into the previous tutorials we have done. I would try the following:

  * insert this `Dropout` module on the input of an autoencoder: that will
  give you a denoising autoencoder

  * now more interesting: insert this `Dropout` module into the convolutional
  network defined in the supervised training tutorial: does it help
  generalization?

Using CUDA and the GPU to Accelerate Training/Testing
-----------------------------------------------------

In Torch, it is (almost) transparent to move parts of your computation graph
to the GPU. 

### Basics: Tensors

The most aggressive way of doing it is to set the default Tensor type to
`Cuda`. Once this is done, any subsequent call to `torch.Tensor` will
allocate a `torch.CudaTensor`. From the Lua interpreter's standpoint, a
`torch.CudaTensor` is really the same as a regular `torch.FloatTensor`. The
only difference is that some operators are not yet implemented, but this
is just a matter of time.

First initialize the environment like this:

```{.lua .numberLines}
require 'cutorch'
torch.setdefaulttensortype('torch.CudaTensor')
print(  cutorch.getDeviceProperties(cutorch.getDevice()) )
```

This should produce something like:

```text
{[deviceOverlap]            = 1
 [textureAlignment]         = 512
 [minor]                    = 0
 [integrated]               = 0
 [major]                    = 2
 [sharedMemPerBlock]        = 49152
 [regsPerBlock]             = 32768
 [computeMode]              = 0
 [multiProcessorCount]      = 16
 [totalConstMem]            = 65536
 [totalGlobalMem]           = 3220897792
 [memPitch]                 = 2147483647
 [maxThreadsPerBlock]       = 1024
 [name]                     = string : "GeForce GTX 580"
 [clockRate]                = 1566000
 [warpSize]                 = 32
 [kernelExecTimeoutEnabled] = 0
 [canMapHostMemory]         = 1}
```

Now you can easily sum two tensors on the GPU by doing this:

```{.lua .numberLines}
t1 = torch.Tensor(100):fill(0.5)
t2 = torch.Tensor(100):fill(1)
t1:add(t2)
```

This summing happened on the GPU.

Now you can very easily move your tensors back and forth the GPU like this:

```{.lua .numberLines}
t1_cpu = t1:float()
t1:zero()
t1[{}] = t1_cpu  -- copies the data back to the GPU, with no new alloc
t1_new = t1_cpu:cuda()  -- allocates a new tensor
```

Knowing this, a more subtle way of working with the GPU is to keep `Tensors`
as `DoubleTensors` or `FloatTensors`, _i.e._ keep them on the CPU by default,
and just move specific `Tensors` to the GPU when needed. We will see that
now when working with `nn`.

### Using the GPU with `nn`

The `nn` module provides modules which each contain their state, and some contain
trainable parameters. When you create a module, the default type of these `Tensors`
is the default type of `Torch`. If you want to create pure `Cuda` modules, then
simply set the default type to `Cuda`, and just create your modules. These modules
will therefore expect `CudaTensors` as inputs. It's often a bit too simplistic
to set things up this way, as your dataset will typically be made of CPU-based 
tensors, and some of your `nn` modules might also be more efficient on the CPU,
such that you'll prefer splitting the model in CPU and GPU pieces.

To use `Cuda`-based `nn` modules, you will need to import `cunn`:

```{.lua .numberLines}
require 'cunn'
```

Assuming that you don't set the default type to `Cuda`, you can easily cast
your modules to any type available in Torch:

```{.lua .numberLines}
-- we define an MLP
mlp = nn.Sequential()
mlp:add(nn.Linear(ninput, 1000))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(1000, 1000))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(1000, 1000))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(1000, noutput))

-- and move it to the GPU:
mlp:cuda()
```

At this stage the network expects `CudaTensor` inputs. Given a `FloatTensor` input,
you will simply need to retype it before feeding it to the model:

```{.lua .numberLines}
-- input
input = torch.randn(ninput)

-- retype and feed to network:
result = mlp:forward( input:cuda() )

-- the result is a CudaTensor, if your loss is CPU-based, then you will
-- need to bring it back:
result_cpu = result:float()
```

Another solution, to completely abstract this issue of type, is to insert `Copy`
layers, which will transparently copy the forward activations, and backward
gradients from one type to another:

```{.lua .numberLines}
-- we put the mlp in a new container:
mlp_auto = nn.Sequential()
mlp_auto:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
mlp_auto:add(mlp)
mlp_auto:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
```

This new `mlp_auto` expects `FloatTensor` inputs and outputs, so you can plug this
guy in any of your existing trainer.

#### Disclaimer/Note

All the basic modules have implemented on CUDA, and all provide
excellent performance. Note though that a lot of modules are still missing, and
we welcome any external help to implement all of them!
