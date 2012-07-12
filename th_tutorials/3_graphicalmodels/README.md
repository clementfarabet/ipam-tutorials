Graphical Models
================

In the previous tutorials, we've learned how to train a family of models
that ranged from _shallow models_ (linear and logistic regression) to
_deep models_ (MLPs, ConvNets). We've seen that however the model was
defined, the overall procedure was always the same:
  
  * define a model, and define a set of trainable/adjustable parameters
  * choose a loss function, an objective to minimize
  * pick an optimization algorithm, to minimize the loss function above,
  by modifying the trainable parameters

These three steps are really general, and can be found in many bodies
of research: statistics, machine learning, computer vision, operating
research, ...

In this tutorial, we're going to explore a different class of models,
which also relies on the procedure above: undirected graphical models,
with pairwise and unary potentials.

To describe graphical models, we use the package `gm`, which can be
installed as any other package: `torch-pkg install gm`. `gm` is
based on [UGM](http://www.di.ens.fr/~mschmidt/Software/UGM.html), 
a really nice Matlab implementation of graphical models. This tutorial
is for the most part based on Mark Schmidt's introduction to UGM and
parameter estimation demos.

Although a bit out of scope in a deep-learning tutorial session,
making the parallel between deep-learning models and graphical models
is of interest. In fact, there are many tasks in signal processing
that benefit from a structured prediction framework, in which a graphical
model imposes a structure on the predictions, and the unary terms of
the model are coming from some classifier. For tasks like scene 
understanding, or OCR, or in general, tasks where there's a need for
jointly segmenting _and_ classifying the input data, the combination
of deep models and graphical models seems worth exploring.

Defining a Model
----------------

In pairwise undirected graphical models, the joint probability of a 
particular assignment to all the variables $x_i$ is represented as
a normalized product of a set of non-negative potential functions:

$$
p(x_1,x_2,...,x_N) = \frac{1}{Z} \prod_{i=1}^{N} \phi_i(x_i) \prod_{e=1}^{E} \phi_e (x_{e_j},x_{e_k})
$$

There is one potential function $\phi$ for each node $i$, for each edge $e$. 
Each edge connects two nodes, $e_j$ and $e_k$. In a full graph, there
is one edge between each possible pair of nodes. For common applications,
such as computer vision, it's much more common to have locally connected
graphs, _i.e._ graphs in which only (small) subsets of nodes are connected
via edges.

The _node potential_ function $\phi_i$ gives a non-negative weight to each possible 
value of the random variable $x_i$. For example, we might set $\phi_i(x_1=0)$ to 0.75 
and $\phi_i(x_1=1)$ to 0.25, which means means that node $1$ has a higher potential 
of being in state $0$ than state $1$. Similarly, the _edge potential_ function $\phi_e$
gives a non-negative weight to all the combinations that $x_{e_j}$ and $x_{e_k}$ can take.

The normalization constant $Z$, or partition function, is a scalar value that forces the
distribution to sum to one, over all possible joint configurations of the variables:

$$
Z = \sum_{x_1}\sum_{x_2}\dots\sum_{x_n} \prod_{i=1}^N \phi_i(x_i) \prod_{e=1}^E \phi_e (x_{e_j},x_{e_k})
$$

This normalizing constant ensures that the model defines a valid probability distribution.

Given the graph structure and the potentials, the `gm` package includes functions for 
four tasks:

  * Decoding: finding the joint configuration of the variables with the highest probability.
  * Inference: computing the normalization constant $Z$, as well as the probabilities of 
  each variable taking each state.
  * Training (or parameter estimation): the task of computing the potentials that maximize the likelihood of set of data.

### A Simple Model

Here we first show how to describe a very simple model. The first
thing to define is the adjacency matrix of the graph, which by itself
fully describes the architecture of the model. The package provides
a couple of standard matrices.

```{.lua .numberLines}
-- define graph
nNodes = 10
adjacency = gm.adjacency.full(nNodes)
```

Once the adjacency matrix is defined, we create a graph, using the `gm.graph`
method. The only information required is the number of possible (discrete)
states per node, and the adjacency matrix. The `maxIter` parameter is used
for loopy graphs, to inform the inference/decoding algorithms on how many
iterations they should go through before stopping.

```{.lua .numberLines}
nStates = 2
g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, verbose=true}
```

In this first example, we are not going to train the potentials, but assume
that we already have them. We need to define two types of potentials, the
$\phi_i$ potentials, and the $\phi_e$ potentials. For each possible configuration,
we need to provide a potential. Potentials are positively correlated
with probabilities, so a high potential means a high probability.

```{.lua .numberLines}
-- unary potentials
nodePot = Tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,1},
                 {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}

-- joint potentials: these are Potts potentials
edgePot = Tensor(g.nEdges,nStates,nStates)
basic = Tensor{{2,1}, {1,2}}
for e = 1,g.nEdges do
  edgePot[e] = basic
end

-- set potentials
g:setPotentials(nodePot,edgePot)
```

At this stage, we have a fully parametrized graphical model, in which
we can do inference and decoding.

But first a few definitions:

  * the _decoding_ task is to find the most likely 
  configuration of the variables.
  * the _inference_ task is to find the normalizing constant $Z$, as well
  as all the marginal probabilities of individual nodes taking each state.

There are multiple methods to do inference and decoding. Exact methods
involve an exhaustive search through all possible combinations of
states.

```{.lua .numberLines}
optimal = g:decode('exact')
print('<gm.testme> maximum likelihood configuration:')
print(optimal)

local nodeBel,edgeBel,logZ = g:infer('exact')
print('<gm.testme> node beliefs:')
print(nodeBel)
print('<gm.testme> edge beliefs:')
print(edgeBel)
print('<gm.testme> log(Z):')
print(logZ)
```

Exact inference is of course impossible for any reasonably-sized model.
A very well known inference technique is _belief propagation_, which 
is also exact in the case of chain-shaped or tree-shaped graphs. In the
case of loopy graphs, the algorithm becomes approximate, and needs to
be run iteratively, until convergence, or until a max number of iterations
has been reached.

Same code using belief propagation:

```{.lua .numberLines}
optimal = g:decode('bp')
print('<gm.testme> optimal config with belief propagation:')
print(optimal)

local nodeBel,edgeBel,logZ = g:infer('bp')
print('<gm.testme> node beliefs:')
print(nodeBel)
print('<gm.testme> edge beliefs:')
print(edgeBel)
print('<gm.testme> log(Z):')
print(logZ)
```

So that's basically all there is to describing arbitrary graphs, thanks
to the adjacency matrix; doing inference; and decoding the optimal joint
configuration of states.

### Conditional Random Field for Images

Let's now move on to a slightly more complex scenario: instead of defining
the potential functions by hand, we're now going to train a set of parameters
to produce _optimal_ potential functions, _i.e._ potential functions that 
maximize the likelihood of each sample.

In this example, we will try to denoise an image, which was corrupted by some
Gaussian noise. In this case, we have input variables (the observed pixels),
and output variables (the clean labels). This is a typical scenario in which
we can use conditional random fields (CRFs).

In CRFs, we have two types of variables: (i) the _features_ $X$ are treated as fixed 
non-random variables (observed), and (ii) the _labels_ $y$ are treated as random variables 
in a graph, where the parameters of the graph depend on the features.

In our case, the _labels_ will be the $32\times32$ clean predictions, which can
take binary states, and the _features_ will be the $32\times32$ pixels, which
are continuous grayscale values.

The first step, as before, is to define the graph structure. We use a 4-connexity
lattice, which is rather typical in the image world:

```{.lua .numberLines}
-- assuming images of geometry:
nRow = 32
nCol = 32
nNodes = nRows*nCols

-- define adjacency matrix (4-connexity lattice)
local adj = gm.adjacency.lattice2d(nRows,nCols,4)

-- create graph
nStates = 2
g = gm.graph{adjacency=adj, nStates=nStates, verbose=true, type='crf', maxIter=10}
```

Note the `type` flag, which is set to `crf`. This defines how the loss function
will be computed. We can now move on to the next section: training this CRF.

Training our Model (CRF)
------------------------

We then need to generate some training data:

```{.lua .numberLines}
-- load a clean, black and white image of an X
sample = image.load('some_32x32_image.png')

-- how many instances to generate in the training data:
nInstances = 100

-- generate some labels (y), which correspond to the MAP solution to the CRF:
y = tensor(nInstances,nRows*nCols)
for i = 1,nInstances do
  y[i] = sample
end
y = y + 1  -- Lua is 1-based

-- generate an ensemble of noisy versions of that image:
X = tensor(nInstances,1,nRows*nCols)
for i = 1,nInstances do
  X[i] = sample
end
X = X + randn(X:size())/2
```

Here are some examples of the noisy input samples:

![](img/noisy.png)

Once we have training data, we need to generate all the _features_ $X$, on which the
CRF is conditioned. As for our previous work in supervised learning, it is key 
to properly whiten / preprocess the training data:

```{.lua .numberLines}
-- create node features (normalized X and a bias)
Xnode = tensor(nInstances,2,nNodes)
Xnode[{ {},1 }] = 1 -- bias
-- normalize features:
nFeatures = X:size(2)
for f = 1,nFeatures do
  local Xf = X[{ {},f }]
  local mu = Xf:mean()
  local sigma = Xf:std()
  Xf:add(-mu):div(sigma)
end
Xnode[{ {},2 }] = X -- features (simple normalized grayscale)
nNodeFeatures = Xnode:size(2)

-- create edge features
nEdges = g.edgeEnds:size(1)
nEdgeFeatures = nNodeFeatures*2-1 -- sharing bias, but not grayscale features
Xedge = zeros(nInstances,nEdgeFeatures,nEdges)
for i = 1,nInstances do
  for e =1,nEdges do
     local n1 = g.edgeEnds[e][1]
     local n2 = g.edgeEnds[e][2]
     for f = 1,nNodeFeatures do
        -- get all features from node1
        Xedge[i][f][e] = Xnode[i][f][n1]
     end
     for f = 1,nNodeFeatures-1 do
        -- get all features from node1, except bias (shared)
        Xedge[i][nNodeFeatures+f][e] = Xnode[i][f+1][n2]
     end
  end
end
```

The last step before training is to decide what the trainable parameters
are going to be. To do so, we create a map that associates trainable
parameters (weights) to nodes and edges in the graph. Here is the
procedure:

```{.lua .numberLines}
-- tie node potentials to parameter vector
nodeMap = zeros(nNodes,nStates,nNodeFeatures)
for f = 1,nNodeFeatures do
  nodeMap[{ {},1,f }] = f
end

-- tie edge potentials to parameter vector
local f = nodeMap:max()
edgeMap = zeros(nEdges,nStates,nStates,nEdgeFeatures)
for ef = 1,nEdgeFeatures do
  edgeMap[{ {},1,1,ef }] = f+ef
  edgeMap[{ {},2,2,ef }] = f+ef
end

-- initialize parameters
g:initParameters(nodeMap,edgeMap)
```

And voila!, we can now train the CRF using arbitrary optimization techniques,
as we did with the MLPs, ConvNets, and autoencoders.

In this case, we minimize the negative log-likelihood in order to maximize the likelihood.
The negative log-likelihood (and its) gradient are computed by the `nll()` method of our
graph (remember that we create the graph with the flag `type='crf'`, which produces
the correct `nll()` method).

With log-linear CRFs, the negative log-likelihood is still differentiable and jointly convex 
in the node and edge parameters, so we could compute its optimal value using a batch algorithm, 
such as L-BFGS. As usual though, if the training data is quite large, starting with an epoch 
or two of SGD can be significantly faster, and then turning to simple averaging SGD might be an 
option, as shown by [Leon Bottou](http://leon.bottou.org/projects/sgd).

```{.lua .numberLines}
-- and train on 30 samples
sgdconf = {learningRate=1e-3}
for iter = 1,100 do
  local i = floor(uniform(1,nInstances)+0.5)
  local feval = function()
     return g:nll(Xnode[i],Xedge[i],y[i],'bp')
  end
  optim.sgd(feval,g.w,sgdconf)
end
```

Note/Exercise: the trainable parameters are availabel in `g.w`. Can you infer how many 
trainable parameters are there in this example? Since you have access to the outer loop
of the SGD trainer, can you easily implement an L2 regularization?

The results of the training procedure are shown here:

```bash
t7> gm.examples.trainCRF()
SGD @ iteration 1: objective = 709.78271289324
SGD @ iteration 2: objective = 86.476069966592
SGD @ iteration 3: objective = 22.048118240666
SGD @ iteration 4: objective = 10.34447115497
SGD @ iteration 5: objective = 12.507434214913
SGD @ iteration 6: objective = -0.24103941715157
SGD @ iteration 7: objective = 13.513625673658
SGD @ iteration 8: objective = 23.680651547275
SGD @ iteration 9: objective = 13.759790868401
SGD @ iteration 10: objective = 24.419772339165
SGD @ iteration 11: objective = 19.331090159347
...
```

This is inference (_i.e._ the marginal probabilities for each pixel):

![](img/inference.png)

And this is the decoding (_i.e._ the optimal joint configuration):

![](img/decoding.png)
