"""
This program shows a pseudo-random colorful square with matplotlib.

(Triple-quotes create block comment regions in Python. Customarily, files begin
with such a block.  It is such an established custom that such comment blocks
are recognized by the interpreter itself as documentation. If you type
help(<module>) then it's this "doc-string" that you see.

"""

from matplotlib import pyplot as plt
from numpy.random import rand, seed

# -- initialize numpy's global random number generator
seed(123)

r = rand(10, 10)

# -- string literals can be either double-quoted or single-quoted
print "r's value as a string is", r
print ' -> and has type', type(r)

# -- add content to the current (default) figure
plt.imshow(r)

# -- show the current figure
plt.show()


