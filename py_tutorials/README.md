
The IPython Deep Learning Tutorials
===================================

The IPython Deep Learning Tutorials are provided as a sequence of IPython
notebooks.


## Launching an IPython notebook server on EC2

We have set up an Amazon EC2 instance (23.23.148.41) for your use during the IPAM summer
school.

To launch the IPython notebook, you need to log in to the instance by ssh.
Our instances does not currently accept password log in, so you'll need the
`ipam_identity.pem` file. Instructions for retrieving this file to your
computer will be given in the class.

Once you've got the `ipam_identity.pem` file,
pick a unique `USERNAME`, and `PORT` for your use within the "student" account,
and if you have a linux or OSX computer type:

    # FIRST TIME LOGIN INSTRUCTIONS
    #
    # connect to our EC2 instance
    ssh -i ipam_identity.pem -L PORT:localhost:PORT student@23.23.148.41

    # set up a workspace for yourself
    mkdir -p USERNAME/.theano
    cd USERNAME
    git clone https://github.com/clementfarabet/ipam-tutorials.git
    cd ipam-tutorials/py_tutorials

    export THEANO_FLAGS=compiledir=~/USERNAME/.theano
    ipython notebook --pylab=inline --port=PORT --no-browser

    # Now point a web browser at http://localhost:PORT to use the tutorials


On subsequent logins, you only need to do the following steps:

    # SUBSEQUENT LOGIN INSTRUCTIONS
    #
    # connect to our EC2 instance
    ssh -i ipam_identity.pem -L PORT:localhost:PORT student@23.23.148.41

    # set up a workspace for yourself
    cd USERNAME/ipam-tutorials/py_tutorials

    export THEANO_FLAGS=compiledir=~/USERNAME/.theano
    ipython notebook --pylab=inline --port=PORT --no-browser

    # Now point a web browser at http://localhost:PORT to use the tutorials
    

When that's all done, point a browser at `http://localhost:PORT` and you should
be looking at a menu of iPython notebooks that we'll be working through this
week.  The notebooks are named like `<D>_<N>_<label>` where `D` is the day we
expect to cover the material, `N` orders the notebooks within a day in
increasing complexity, and `label` gives the topic of the notebook.

There is more material in each day `D` than you will be able to cover within
the fraction of an hour allotted to the practical sessions, but they should be
relatively stand-alone documents, and you are encouraged to come back to them
and work through them at your own pace.

In the interest of keeping memory usage under control you should make sure to
*shut down* the notebooks that you are not using by pressing that notebook's
"Shutdown" button on the "IPython Dashboard".

## Updating and Sychronizing your files with `git`

While you work in the IPython notebooks, you might want to *save* your changes
so that they will still be there after you log out.

If we make changes to the notebooks, we will update the central
github repository, and you will have the *choice* of whether you want to use the
new versions or not. If you want to use the new versions type:

    cd ~/USERNAME/ipam-tutorials
    git commit -am 'Saving changes made by USERNAME'
    git pull

This will update your local files with the ones from github.
If you have saved changes to the same parts of the same files that we have
modified, then a version conflict may occur. The `git` revision control software
allows you to handle this in several ways. You can either (a) ask for assistance
at this point, or (b) [resolve the
conflict](http://genomewiki.ucsc.edu/index.php/Resolving_merge_conflicts_in_Git),
or (c) forget about merging the file versions from github (for the time being) by
typing `git reset --hard`.


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

