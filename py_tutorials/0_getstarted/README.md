
Getting Started with Scientific Python
======================================


Hello Random World
------------------

Python is interpreted, so you can just type `python` to start it up and start
typing statements, function definitions, etc. at the prompt.  To run
statements etc. that are in a file, type `python <filename>.py`. To try things
out interactively, you can use the `ipython` shell.  Type `ipython -pylab`.

    >>> imshow(rand(10, 10))

If things are set up right, you'll see a bunch of colors in a square.  The
`-pylab` option turns ipython into something like MATLAB. Lots of symbols are
defined at startup that are not normally built in to Python.

To get more of a sense of how to do this same thing in normal Python, quit
ipython (CTRL-D), and start it again with no options: `ipython`.  To get the
same effect we have to import the symbols explicitly:

    >>> from matplotlib import pyplot as plt
    >>> from numpy.random import rand
    >>> plt.imshow(rand(10, 10))
    >>> plt.show()


This syntax (that works in `ipython`) is pure Python that we can execute from a file using the normal Python interpreter.
If you cut and paste these lines into a file with a `.py` suffix, then you can
execute it by typing `python <filename>.py` and you should see another random
box appear.

    from matplotlib import pyplot as plt
    from numpy.random import rand
    plt.imshow(rand(10, 10))
    plt.show()


N.B. If you look closely, you'll see that a *different random pattern* appears
each time you run this program.  This is kinda fun, but generally not what you
want when you're trying to do science. You generally want to get the *same
random numbers every time*, so that you can reproduce your computation.  So
get in the habit of *seeding* your generators right off the bat like this:

    from matplotlib import pyplot as plt
    from numpy.random import rand, seed

    seed(123)

    plt.imshow(rand(10, 10))
    plt.show()


This example is already saved in the accompanying [hello_random.py](./0_getstarted/hello_random.py), so you can just type `python hello_random.py` in the current directory to try it out.


Browing through CIFAR10
-----------------------

The data sets we will use in the upcoming tutorials are provided via the
[skdata][skdata] package. The skdata package provides the logic of downloading,
unpacking, and providing a meaningful Python interface to various public data
sets.

Python has a tradition of "batteries included" design (have a look at the
[standard library]() for example, it includes many file-loading routines and even a
[web server]()!)
In keeping with that tradition, [skdata][skdata] datasets contain little main
scripts with sub-commands that do standard dataset-related things (and sometimes
more specific dataset-related things.)  For example, if `glumpy` and `skdata` have
been installed, then the following script should launch a little program to view
the images in CIFAR10:

    $ python -m skdata.cifar10.main show

The 'j' and 'k' keys step forward and back through the set of images and the 'q'
key quits.

    $ python -m skdata.mnist.main show
    $ python -m skdata.svhn.main show  # XXX

The `python -m` command executes an importable module as if you had run it as a
script like we did in the examples above.  The `python -c` command runs a
string as if it were the contents of a script, so we can run from the bash
shell:

    $ python -c 'import skdata.mnist.main; print skdata.mnist.main.__file__'

This command prints out the location in the filesystem of the main.py file that
we ran. Open that file up with your favorite text editor to get a sense for how
to use skdata. (HINT: if the __file__ ends with '.pyc' then the interpreter has
loaded the pre-parsed bytecode version. There is a '.py' file next to it that
you can read.)

[skdata]: http://jaberg.github.com/skdata/ "Scikit-data"
  

Python Software Ecosystem
-------------------------

Deep learning can take your research in many directions, and it's nice that
Python has a lot of projects to help you on your way:

* [Python](http://python.org/) - the standard library is surprisingly comprehensive, get to
  know what's in it.

* [NumPy](http://numpy.scipy.org/) - the defacto official n-dimensional array data type for Python.
  Doing numerical work in Python, you will use these *all the time*.

* [SciPy](http://www.scipy.org) - a library of many algorithms and utilities
  that are useful across a broad range of scientific work.
  I originally captioned the following "especially useful" but the list grew to
  include almost the entire library:

  * [scipy.io](http://docs.scipy.org/doc/scipy/reference/io.html) - read and write various matrix formats including MATLAB

  * [scipy.linalg](http://docs.scipy.org/doc/scipy/reference/linalg.html) - decompositions, inverses, etc.

  * [scipy.ndimage](http://docs.scipy.org/doc/scipy/reference/ndimage.html) - basic image processing

  * [scipy.optimize](http://docs.scipy.org/doc/scipy/reference/optimize.html) - 1-D and N-D optimization algorithms

  * [scipy.signal](http://docs.scipy.org/doc/scipy/reference/signal.html) - signal processing

  * [scipy.sparse](http://docs.scipy.org/doc/scipy/reference/sparse.html) - several sparse matrix types

  * [scipy.special](http://docs.scipy.org/doc/scipy/reference/special.html) - special functions (e.g. erf)
  
  * [scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html) - pdfs, cdfs, sampling algorithms, tests 

* [IPython](http://ipython.org/) is a feature-rich Python interpreter.
  Personally I haven't really gotten into it, but lots of people love it and
  it can do a lot of amazing things (incl. distributed processing and
  collaborative html workspace publishing).

* [Matplotlib](http://matplotlib.sourceforge.net/) is the most widely-used plotting library for Python. It provides an
  interface similar to that of MATLAB or R. It is sometimes annoying, and not as flashy as say,
  [d3](http://d3js.org/) but it is the workhorse of data visualization in Python.

* [Theano](http://www.deeplearning.net/software/theano/) - an array expression compiler, with automatic differentiation and behind-the-scenes GPU execution.

* [PyAutoDiff](https://github.com/jaberg/pyautodiff) - provides automatic differentiation for code using NumPy.
  Currently this is a Theano front-end, but the API has been designed to keep Theano out of the sight of client code.

* [pandas](http://pandas.pydata.org/) - machine learning algorithms and types, mainly for working with time series.
  Includes R-like data structures (like R's data frame).

* [Cython](http://cython.org/) - a compiler for a more strongly-typed Python dialect, very useful for optimizing numeric Python code.

* [Copperhead](http://copperhead.github.com) - compiles natural python syntax to GPU kernels

* [numexpr](http://code.google.com/p/numexpr/) - compiles expression strings to perform loop-fused element-wise computations

* [scikit-learn](http://scikit-learn.org/stable/) - well-documented and well-tested implementations of many standard ML algorithms.

* [scikits-image](http://scikits-image.org/) - image-processing (edge detection, color spaces, many standard algorithms)

* [skdata](http://jaberg.github.com/skdata) - data sets for machine learning



Scientific Python Tutorials
---------------------------

There are several great resources to help you get oriented to use Python for
numeric scientific work.

* [Unix/Python/NumPy](http://www.cs.utah.edu/~hal/courses/2009F_ML/p0/) - This
  is a really high-level intro to unix and Python.  There is tons of
  documentation for these things and they're kind of out of scope of these
  tutorials--by all means dig around the net for more background if you want
  to know more.

* [NumPy basics](http://www.scipy.org/Tentative_NumPy_Tutorial) 

* [SciPy Getting Started](http://www.scipy.org/Getting_Started)

* [Deep Learning Tutorials](http://www.deeplearning.net/tutorial/) provide code-centered introductions to 
  Deep Learning algorithms and architectures.
  They make heavy use of Theano, and illustrate good practices
  of programming with Theano.

