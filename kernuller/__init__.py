import numpy as np

from . import kernel_cm
from .kernel_cm import br, bo, vdg, bbr
#from .detection_maps import detection_maps as dmap


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
from tqdm import tqdm

import astropy
import astropy.units as u
import copy

from itertools import combinations, permutations

from . import kernuller
from .kernuller import *



version_info = (0,1,0)
__version__ = '.'.join(str(c) for c in version_info)

# -------------------------------------------------
# set some defaults to display images that will
# look more like the DS9 display....
# -------------------------------------------------
#plt.set_cmap(cm.gray)
(plt.rcParams)['image.origin']        = 'lower'
(plt.rcParams)['image.interpolation'] = 'nearest'
# -------------------------------------------------

plt.ion()
plt.show()
