
The IPython Deep Learning Tutorials
===================================

The IPython Deep Learning Tutorials are provided as a sequence of IPython
notebooks.


## Launching an IPython notebook server on EC2

We have set up an Amazon EC2 instance for your use during the IPAM summer
school.
The IP number of the instance changes periodically, so we'll just tell you
what it is when you need it.

To launch the IPython notebook, you need to log in to the instance by ssh.
Our instances does not currently accept password log in, so you'll need the
`ipam_identity.pem` file. Instructions for retrieving this file to your
computer will be given in the class.

Once you've got the `ipam_identity.pem` file and the `ADDRESS` of the server,
pick a unique `USERNAME`, and `PORT` for your use within the "student" account,
and if you have a linux or OSX computer type:

    # connect to our EC2 instance "ADDRESS"
    ssh -i ipam_identity.pem -L 8889:localhost:8889 student@ADDRESS

    # set up a workspace for yourself
    mkdir -p USERNAME/.theano
    cd USERNAME
    git clone https://github.com/clementfarabet/ipam-tutorials.git
    cd ipam-tutorials/py_tutorials

    export THEANO_FLAGS=compiledir=~/USERNAME/.theano
    ipython notebook --pylab=inline --port=PORT


When that's all done, point a browser at `http://localhost:PORT` and you should
be looking at a menu of iPython notebooks that we'll be working through this
week.  The notebooks are named like `<D>_<N>_<label>` where `D` is the day we
expect to cover the material, `N` orders the notebooks within a day in
increasing complexity, and `label` gives the topic of the notebook.

There is more material in each day `D` than you will be able to cover within
the fraction of an hour allotted to the practical sessions, but they should be
relatively stand-alone documents, and you are encouraged to come back to them
and work through them at your own pace.



### Windows instructions

I don't know how to run ssh from windows. If you can do it, please let us know
now! If you also do not know how, ask your neighbour to run the steps above
for you. Once you have an ipython notebook running, you interact with it
directly through your browser anyway.


## Installation and Requirements

If you want to set up these tutorials to run on your laptop or another
computer, you'll need a fairly standard scientific Python stack,
and you'll also need to install a few
non-standard project versions of

* Theano
* pyautodiff
* skdata

Working versions are available in this repo's `py_submodules` directory as git submodules.

Installing these things is not yet the turn-key process that Python users have
come to expect.  Let us know after class if you want to set these things up
and we can work through it during off hours.

