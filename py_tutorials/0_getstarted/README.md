
Getting Started with Scientific Python
======================================

There are several great resources to help you get oriented to use Python for
numeric scientific work.

* [Unix/Python/NumPy](http://www.cs.utah.edu/~hal/courses/2009F_ML/p0/) - This
  is a really high-level intro to unix and Python.  There is tons of
  documentation for these things and they're kind of out of scope of these
  tutorials--by all means dig around the net for more background if you want
  to know more.

* [NumPy basics](http://www.scipy.org/Tentative_NumPy_Tutorial) - [NumPy]() is
  the defacto official n-dimensional array data type for Python. Doing
  numerical work in Python, you will use these *all the time*.

* [SciPy Getting Started](http://www.scipy.org/Getting_Started) - This is a
  point of entry for `scipy` which is a core package with lots of useful
  things, whose contributors (together with the NumPy people) form the core
  of the scientific Python community.

* [IPython](http://ipython.org/) is a feature-rich Python interpreter.
  Personally I haven't really gotten into it, but lots of people love it and
  it can do a lot of amazing things (incl. distributed processing and
  collaborative html workspace publishing).


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

