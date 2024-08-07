import numpy as np

from . import kernel_cm
from .kernel_cm import br, bo, vdg, bbr
from . import interferometers
from .interferometers import (CHARA,
                            VLTI,
                            redarray,
                            VLTI_AT_small,
                            VLTI_AT_medium,
                            VLTI_AT_large,
                            VLTI_AT_astrometric,
                            VLTI_AT_small5,
                            VLTI_AT_medium5,
                            VLTI_AT_large5,
                            VLTI_AT_astrometric5,
                            VLTI_AT_small6,
                            VLTI_AT_medium6,
                            VLTI_AT_large6,
                            VLTI_AT_astrometric6)
from . import diagrams
from .diagrams import plot_outputs_smart as cmp
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
