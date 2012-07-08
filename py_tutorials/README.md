
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
    # XXX point user to ~james/.theanorc
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

