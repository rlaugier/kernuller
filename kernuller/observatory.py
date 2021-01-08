
#print("Loading observatory tool")

import numpy as np
import sympy as sp


from astropy.time import Time
import astropy.units as u

import astroplan
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun

"""
Basic usage:

import kernuller.observatory
myobs = kernuller.observatory.observatory(kernuller.VLTI)
tarnames = "Spica"
targets = [kernuller.observatory.astroplan.FixedTarget.from_name(tar) for tar in tarnames]
obstimes = myobs.build_observing_sequence()
target_positions = myobs.get_positions(targets[0], obstimes)
newarray = myobs.get_projected_array(myobs.get_positions(targets, obstimes)[0,0])
"""

class observatory(object):
    """
    This class help define the properties of the observatory infrastructure, especially the uv coverage.
    """
    def __init__(self, statlocs, location=None, verbose=False):
        """
        statlocs : The station locations 
                    (east, north) for each aperture shape is (Na, 2)
        location : An astropy.coordinatEarthLocation (default = Paranal)
                    example: myloc = astroplan.Observer.at_site("Paranal", timezone="UTC")
        """
        self.verbose = verbose
        if location is None:
            self.observatory_location = astroplan.Observer.at_site("Paranal", timezone="UTC")
        else :
            self.observatory_location = location
        self.statlocs = statlocs
        
        self.theta = sp.symbols("self.theta")
        #R handles the azimuthal rotation
        self.Rs = sp.Matrix([[sp.cos(self.theta), sp.sin(self.theta)],
                       [-sp.sin(self.theta), sp.cos(self.theta)]])
        self.R = sp.lambdify(self.theta, self.Rs, modules="numpy")
        #P handles the projection due to elevation rotation
        self.Ps = sp.Matrix([[1, 0],
                       [0, sp.sin(self.theta)]])
        self.P = sp.lambdify(self.theta, self.Ps, modules="numpy")
        #C handles the piston due to elevation.
        self.Cs = sp.Matrix([[1, sp.cos(self.theta)]])
        self.C = sp.lambdify(self.theta, self.Cs, modules="numpy")

        
    def build_observing_sequence(self, times=["2020-04-13T00:00:00","2020-04-13T10:30:00"],
                            npoints=20, remove_daytime=False):
        """
        Returns the series of obstimes needed to compute the altaz positions
        times : a list of UTC time strings ("2020-04-13T00:00:00")
                that define an interval (if npoints is not None,
                or the complete list of times (if npoints is None)
        npoints : The number of samples to take on the interval
                 None means that the times is the whole list
                 
        remove_daytime : Whether to remove the points that fall during the day
        """
        #npoints is defined which means we work from define the sampling from an interval
        if npoints is not None:
            obs2502 = Time(times)
            dt = obs2502[1] - obs2502[0]
            obstimes = obs2502[0] + dt * np.linspace(0.,1., npoints)
        #npoints is None means the times represent a list of times 
        else: 
            obstimes = np.array([Time(times[i]) for i in range(len(obstimes))])
            
        totaltime = (obstimes[-1]-obstimes[0]).to(u.s).value
        if remove_daytime:
            totaltime = (obstimes[-1]-obstimes[0]).to(u.s).value
            halfhourpoints = int(npoints / (totaltime / 900))
            totaltime = (obstimes[-1]-obstimes[0]).to(u.s)
            mask = observatory_location.sun_altaz(obstimes).alt<0
            sunelev = observatory_location.sun_altaz(obstimes).alt
            rawsunelev = np.array([el.value for el in sunelev])
            midnight = np.argmin(rawsunelev)
            sunrise = np.argmin(np.abs(rawsunelev))
        return obstimes
            
        
    def get_positions(self, targets, obstimes):
        """Returns the astropy.coordinates.AltAz for a given target
        targets: A list of SkyCoord objects 
        obstimes: A list of astropy.Times to make the observations
        """
        taraltaz = self.observatory_location.altaz(target=targets,
                                                   time=obstimes,
                                                   grid_times_targets=True)
        return taraltaz
        
    def get_projected_array(self, taraltaz):
        """
        Returns the new coordinates for the projected array
        taraltaz : the astropy.coordinates.AltAz of the target
        """
        arrayaz = self.R((taraltaz.az.value - 180)*np.pi/180).dot(self.statlocs.T).T
        newarray = self.P(taraltaz.alt.value * np.pi/180).dot(arrayaz.T).T
        if self.verbose:
            print("=== AltAz position:")
            print("az", taraltaz.az.value -180)
            print("alt", taraltaz.alt.value)
            print("old array", self.statlocs)
            pritn("new array", newarray)
        return newarray
    def get_projected_geometric_pistons(self, taraltaz):
        """
        Returns the geomtric piston resutling from the pointing
        of the array.
        taraltaz : the astropy.coordinates.AltAz of the target
        """
        arrayaz = self.R((taraltaz.az.value - 180)*np.pi/180).dot(self.statlocs.T).T
        pistons = self.R(taraltaz.alt.value * np.pi/180).dot(arrayaz.T).T
        if self.verbose:
            print("=== AltAz position:")
            print("az", taraltaz.az.value -180)
            print("alt", taraltaz.alt.value)
            print("old array", self.statlocs)
            pritn("new array", pistons)
        return newarray