
The IPython Deep Learning Tutorials
===================================

On the Amazon EC2 instance we've set up, pick a `USERNAME` and a `PORT` and type:

    # connect to our EC2 instance "HOSTNAME"
    ssh -L 8889:localhost:8889 student@HOSTNAME

    # set up a workspace for yourself
    mkdir USERNAME
    mkdir USERNAME/.theano
    cd USERNAME
    git clone https://github.com/clementfarabet/ipam-tutorials.git
    cd ipam-tutorials/ipy

    # launch the ipython notebook
    export THEANO_FLAGS=compiledir=~/USERNAME/.theano
    ipython notebook --pylab=inline --port=PORT


If that's all done, point a browser at `http://localhost:PORT` and you should
be looking at a menu of iPython notebooks that we'll be working through this
week.


## Installation and Requirements

Several non-standard project versions are required to run these IPython
tutorials:

* Theano
* pyautodiff
* skdata

Working versions are available in this repo's `py_submodules` directory as git submodules.


## Notational Conventions:

* Expressions that appear in fixed-width font like `a = b + c` are meant to be
  interpreted as Python code, whereas expressions appearing in italics like $a
  = b + c$ are meant to be interpreted as math notation.

* In math notation, non-bold lower-case symols denote scalars.

* In math notation, bold lower-case symbols like $\mathbf{x}$ and
  $\mathbf{h}$ denote vectors.

* In math notation, upper-case symbols like $W$ and $V$ typically denote
  matrices.

* In the LaTeX-powered "math" expressions the notation $[a, b]$ denotes
  the _continuous inclusive range_ of real-valued numbers $u$ satisfying
  $0 \leq x \leq 1$.  In contrast, the Python syntax `[a, b]` denotes a
  _list_ data structure of two elements.

* Python and NumPy tend to favor C-style "row-major" matrices, and we
  will follow that convention in the math sections too.
  Consequently, vector-matrix products will typically be written with
  the vector on the _left_ and the matrix on the _right_, as in
  $\mathbf{h}V$.
