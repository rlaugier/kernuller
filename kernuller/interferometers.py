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
#VLTI_AT_small = np.load(resource_path.joinpath("vlti_at_small.npy"))[:,:-1]
#VLTI_AT_medium = np.load(resource_path.joinpath("vlti_at_medium.npy"))[:,:-1]
#VLTI_AT_large = np.load(resource_path.joinpath("vlti_at_large.npy"))[:,:-1]

# From the UTILS by Antoine Mérand
layout_orientation = -18.984 # degrees

#      station   p        q        E        N       A0
vlti_layout = {'A0':(-32.0010, -48.0130, -14.6416, -55.8116, 129.8495),
                'A1':(-32.0010, -64.0210, -9.4342, -70.9489, 150.8475),
                'B0':(-23.9910, -48.0190, -7.0653, -53.2116, 126.8355),
                'B1':(-23.9910, -64.0110, -1.8631, -68.3338, 142.8275),
                'B2':(-23.9910, -72.0110, 0.7394, -75.8987, 158.8455),
                'B3':(-23.9910, -80.0290, 3.3476, -83.4805, 166.8295),
                'B4':(-23.9910, -88.0130, 5.9449, -91.0303, 150.8275),
                'B5':(-23.9910, -96.0120, 8.5470, -98.5942, 174.8285),
                'C0':(-16.0020, -48.0130, 0.4872, -50.6071, 118.8405),
                'C1':(-16.0020, -64.0110, 5.6914, -65.7349, 142.8465),
                'C2':(-16.0020, -72.0190, 8.2964, -73.3074, 134.8385),
                'C3':(-16.0020, -80.0100, 10.8959, -80.8637, 150.8375),
                'D0':(0.0100, -48.0120, 15.6280, -45.3973, 97.8375),
                'D1':(0.0100, -80.0150, 26.0387, -75.6597, 134.8305),
                'D2':(0.0100, -96.0120, 31.2426, -90.7866, 150.8275),
                'E0':(16.0110, -48.0160, 30.7600, -40.1959, 81.8405),
                'G0':(32.0170, -48.0172, 45.8958, -34.9903, 65.8357),
                'G1':(32.0200, -112.0100, 66.7157, -95.5015, 129.8255),
                'G2':(31.9950, -24.0030, 38.0630, -12.2894, 73.0153),
                'H0':(64.0150, -48.0070, 76.1501, -24.5715, 58.1953),
                'I1':(72.0010, -87.9970, 96.7106, -59.7886, 111.1613),
                'J1':(88.0160, -71.9920, 106.6481, -39.4443, 111.1713),
                'J2':(88.0160, -96.0050, 114.4596, -62.1513, 135.1843),
                'J3':(88.0160, 7.9960, 80.6276, 36.1931, 124.4875),
                'J4':(88.0160, 23.9930, 75.4237, 51.3200, 140.4845),
                'J5':(88.0160, 47.9870, 67.6184, 74.0089, 164.4785),
                'J6':(88.0160, 71.9900, 59.8101, 96.7064, 188.4815),
                'K0':(96.0020, -48.0060, 106.3969, -14.1651, 90.1813),
                'L0':(104.0210, -47.9980, 113.9772, -11.5489, 103.1823),
                'M0':(112.0130, -48.0000, 121.5351, -8.9510, 111.1763),
                'U1':(-16.0000, -16.0000, -9.9249, -20.3346, 189.0572),
                'U2':(24.0000, 24.0000, 14.8873, 30.5019, 190.5572),
                'U3':(64.0000, 48.0000, 44.9044, 66.2087, 199.7447),
                'U4':(112.0000, 8.0000, 103.3058, 43.9989, 209.2302)}

def get_list2layout(conflist):
    statlocs = np.array([vlti_layout[thekey][2:4] for thekey in conflist])
    return statlocs
VLTI_AT_large = get_list2layout(("A0", "G1", "J2", "J3"))
VLTI_AT_medium = get_list2layout(("K0", "G2", "D0", "J3"))
VLTI_AT_small = get_list2layout(("A0", "B2", "D0", "C1"))
VLTI_AT_astrometric = get_list2layout(("A0", "G1", "J2", "K0"))

VLTI_AT_large5 = get_list2layout(("A0", "G1", "J2", "J3", "M0"))
VLTI_AT_medium5 = get_list2layout(("K0", "G2", "D0", "J3", "H0"))
VLTI_AT_small5 = get_list2layout(("A0", "B2", "D0", "C1", "E0"))
VLTI_AT_astrometric5 = get_list2layout(("A0", "G1", "J2", "K0", "B5"))

VLTI_AT_large6 = get_list2layout(("A0", "G1", "J2", "J3", "M0", "B5"))
VLTI_AT_medium6 = get_list2layout(("K0", "G2", "D0", "J3", "H0", "J1"))
#VLTI_AT_small6 = get_list2layout(("A0", "B2", "D0", "C3", "E0", "G2"))
VLTI_AT_small6 = get_list2layout(("A0", "B2", "D1", "C2", "E0", "G0"))
VLTI_AT_astrometric6 = get_list2layout(("A0", "G1", "J2", "K0", "B5", "E0"))


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

