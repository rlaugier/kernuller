import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

import astropy
import astropy.units as u
import astropy.coordinates
from pathlib import Path

resource_path = Path(__file__).parent / "data"

#VLTI layout

VLTI_stations = list(["U1", "U2", "U3", "U4"])
VLTI = np.array([[-9.925,-20.335],
                    [14.887,30.502],
                    [44.915,66.183],
                    [103.306,44.999]])
VLTI_AT_small = np.load(resource_path.joinpath("vlti_at_small.npy"))[:,:-1]
VLTI_AT_medium = np.load(resource_path.joinpath("vlti_at_medium.npy"))[:,:-1]
VLTI_AT_large = np.load(resource_path.joinpath("vlti_at_large.npy"))[:,:-1]

#A redundant array

redarray = np.array([[0,0],
                    [0,100],
                    [0,200],
                    [100,0],
                    [100,100],
                    [100,200]])

#The CHARA array.
telcoords = resource_path.joinpath("chara.dat").read_text()
#telcoords = open(__name__).read()
telcoords = telcoords.split("\n\n")
stations = [telcoords[i].split("\n")[0] for i in range(len(telcoords))]
longitudes = [telcoords[i].split("\n")[1].split('  ')[1] for i in range(len(telcoords))]
latitudes = [telcoords[i].split("\n")[2].split('  ')[1] for i in range(len(telcoords))]
statearthlocs = []
for i in range(len(longitudes)):
    statearthlocs.append(astropy.coordinates.EarthLocation(lon=longitudes[i], lat=latitudes[i], height=1740))
a = statearthlocs[0]
radius = np.sqrt(a.x**2 + a.y**2 + a.z**2)
N = [((statearthlocs[i].lat - statearthlocs[0].lat).to(u.rad) * radius).value for i in range(len(statearthlocs))]
E = [((statearthlocs[i].lon - statearthlocs[0].lon).to(u.rad) * radius).value for i in range(len(statearthlocs))]
CHARA = np.array([N,E]).T

