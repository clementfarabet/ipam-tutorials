
from skdata import cifar10
n_examples = 10000
# -- RECOMMENDED: restart the IPython kernel to clear out memory
data_view = cifar10.view.OfficialImageClassification(x_dtype='float32', n_train=n_examples)
x = data_view.train.x[:n_examples].transpose(0, 3, 1, 2)


import numpy as np
import util
patches = util.random_patches(x, 1000, 7, 7, np.random)

print patches
