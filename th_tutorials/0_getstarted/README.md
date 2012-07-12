Getting Started with Torch
==========================

Torch7 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. It is easy to use and provides a very efficient 
implementation, thanks to an easy and fast scripting language,
[Lua](http://www.lua.org/) and an underlying C/OpenMP/CUDA implementation.

Torch7 is developed at 
[Idiap Research Institute](http://www.idiap.ch/), 
[New York University](http://www.cs.nyu.edu/~yann/) and
[NEC Laboratories America](http://www.nec-labs.com/). 

At this time, Torch7 is maintained by 
[Ronan Collobert](http://ronan.collobert.com/), 
[Clement Farabet](http://www.clement.farabet.net/)
and 
[Koray Kavukcuoglu](http://koray.kavukcuoglu.org/).
We also use the Qt <-> Lua interface from 
[Leon Bottou](http://leon.bottou.org/).

Installing Torch
----------------

For these tutorials, we assume that Torch is already installed, as well as extra
packages (_image_, _nnx_, _unsup_). If you want to install Torch on your machine
(which I recommend for the next sessions), follow the instructions available
[here](http://www.torch.ch/manual/install/index). A more condensed version can 
also be found [here](http://code.cogbits.com/).

EC2 Machine: Pre-built Torch for the tutorials
----------------------------------------------

For now, we have set up a server (Amazon EC2), which contains a pre-built version 
of Torch, and all the packages necessary to run the demos/tutorials provided.
The IP number of the server will change periodically, so we'll just tell you
what it is when you need it.

We will provide you with an identity file `ipam_identity.pem` file, which you'll use
to log into the EC2 machine. Once you've got the `ipam_identity.pem` file and the 
`ADDRESS` of the server, pick a unique `USERNAME`, for your use within the "student"
account, and if you have a linux or OSX computer type:

```{.bash .numberLines}
# connect to our EC2 instance "ADDRESS"
ssh -XC -i ipam_identity student@23.23.148.41

# it's important to use the -XC flags, to enable graphics

# set up a workspace for yourself
mkdir -p USERNAME/
cd USERNAME
git clone https://github.com/clementfarabet/ipam-tutorials.git
cd ipam-tutorials/th_tutorials
```

At this stage, you have all the code for the tutorials in this directory.
For the first day, we will start with the code in `0_getstarted/`, then
on the second day we will attack `1_supervised/`, and so on.

Running Code
------------

Torch7 is interpreted, so the easiest way to get started is to start an
interpreter, and start typing commands:

```{.bash .numberLines}
$ torch
Try the IDE: torch -ide
Type help() for more info
Torch 7.0  Copyright (C) 2001-2011 Idiap, NEC Labs, NYU
Lua 5.1  Copyright (C) 1994-2008 Lua.org, PUC-Rio
```

```{.lua .numberLines}
t7> a = 10
t7> =a
10
t7> print 'something'
something
t7> = 'something'
something
t7> 
```

By default, the interpreter preloads the torch and plotting packages. Extra
packages, such as image, and nn, must be required manually:

```{.lua .numberLines}
t7> require 'nn'
t7> require 'image'
```

The image package allows easy rendering of images:

```{.lua .numberLines}
t7> i = image.lena()
t7> image.display(i)
```

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/0_getstarted/img/lena.png)

In these tutorials, we'll be interested in visualizing internal states, and
convolution filters:

```{.lua .numberLines}
t7> n = nn.SpatialConvolution(1,16,12,12)
t7> image.display{image=n.weight, padding=2, zoom=4}
```

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/0_getstarted/img/filters.png)

```{.lua .numberLines}
t7> n:forward(image.rgb2y(i))
t7> image.display{image=n.output, padding=2, zoom=0.25, legend='states'}
```

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/0_getstarted/img/states.png)

Note: most functions in Torch support named arguments. In the example above, 
image.display can be called with arguments in order, or named. When using named
arguments, notice that we replace the parenthesis by curly brackets. Curly
brackets are used to describe Lua's most general data type: the table.

A more advanced, graphical interpreter can be launched like this:

```{.bash .numberLines}
$ torch -ide
```

![](https://github.com/clementfarabet/ipam-tutorials/raw/master/th_tutorials/0_getstarted/img/ide.png)

This console has better support for history, and completion.

Last, but not least, you can of course save code to a file, which should use
the .lua extension, and execute it either from your shell prompt, or from
the Torch interpreter. All the code in this file has been written to a file
called [getstarted.lua](./getstarted.lua). You can run it like this:

```{.bash .numberLines}
$ torch getstarted.lua
```

or from the Torch interpreter:

```{.lua .numberLines}
t7> dofile 'getstarted.lua'
```

Getting help
------------

Torch's documentation is work in progress, but you can already get help for most function 
provided in the official packages (_torch_, _nn_, _gnuplot_, _image_). Documentation
is installed on your machine, along with Torch's packages. The default location is 
[/usr/local/share/torch/html/index.html](file:///usr/local/share/torch/html/index.html). It
is also mirrored online [here](http://www.torch.ch/manual), and can be triggered from
your interpreter by doing:

```{.lua .numberLines}
t7> browse()
```

Now a quick way to learn functions, and explore packages, is to use TAB-completion, that is,
start typing something in the Torch interpreter, and then hit TAB twice:

```{.lua .numberLines}
t7> torch. + TAB
torch.ByteTensor.            torch.include(
torch.CharStorage.           torch.initialSeed(
torch.CharTensor.            torch.inverse(
...
```

This will give you a list of all functions provided in the _torch_ package. You can then
get inline help for a specific function like this:

```{.lua .numberLines}
t7> help(torch.randn)

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[res] torch.randn( [res,] m [, n, k, ...])       
 y=torch.randn(n) returns a one-dimensional tensor of size n filled 
with random numbers from a normal distribution with mean zero and variance 
one.
 y=torch.randn(m,n) returns a mxn tensor of random numbers from a normal 
distribution with mean zero and variance one.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
```

Going Further
-------------

Now that you have all the basic pieces, I invite you to take a look at Torch's
[basic tutorial/introduction](http://www.torch.ch/manual/tutorial/index). In
particular, take a look at: 

  * a very quick [Lua primer](http://www.torch.ch/manual/tutorial/index#lua_basics),
  * a fast intro to [Torch types](http://www.torch.ch/manual/tutorial/index#torch_basicsplaying_with_tensors),
  * a very simple introduction to [neural network training](http://www.torch.ch/manual/tutorial/index#exampletraining_a_neural_network).

The Torch core is a general numeric library, and although most early packages were
built around neural network training, lots of 3rd packages have developed that
provide code for all sorts of things, including image processing, video processing,
graphical models (CRFs, ...), parallel computing, camera interfaces, ... A complete
list of packages we support can be found [here](http://code.cogbits.com/packages/).
Torch has a built-in package management system that makes it very easy for anyone
to get and develop new packages, which can be shared easily, using, for example 
[GitHub](https://github.com/) as a distribution platform. More details about this
[here](http://www.torch.ch/manual/install/index#the_torch_package_management_system).

(Easy) Exercise
---------------

Ok it's day 1, we just got started, but if we have enough time, let's try to anticipate 
what we'll have to do in the next days, with a simple exercise.

In any machine learning task, data plays a central role. The most basic thing to
do when getting new data is to ensure that it is properly normalized. This is
valid for any ML problem. You will find the MNIST dataset already stored into
the Torch file format [here](http://data.neuflow.org/data/mnist.t7.tgz). You can
download it like this:

```{.bash .numberLines}
$ wget http://data.neuflow.org/data/mnist.t7.tgz
$ tar xvf mnist.t7.tgz
```

You can then load both the training and test data like this:

```{.lua .numberLines}
t7> train = torch.load('mnist.t7/train_32x32.t7', 'ascii')
t7> test = torch.load('mnist.t7/test_32x32.t7', 'ascii')
t7> = train
{[data]   = ByteTensor - size: 60000x1x32x32
 [labels] = ByteTensor - size: 60000}
```
	
Data needs to be normalized. Verify the initial data range, and
write the necessary code to insure that the data has zero mean and
unit norm. Depending on the task, you might want to normalize the
data globally (at the level of the training set) or independently
for each patch, or for each feature (pixel in this case).

You will need to be able to slice these arrays, to make use of
efficient numeric routines (ala Matlab). If you can't figure out how
to do it, check out this script: `../1_supervised/A_slicing.lua`.

Last little challenge, try to display a subset of the training
images. By now you should have seen enough stuff to be able to 
do that.
