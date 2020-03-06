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


coeff = 1/sp.sqrt(2)
def ph_shifter(phi):
    return sp.exp(sp.I*phi)
    
splitter = sp.Matrix([[coeff],
                     [coeff]])
crossover = sp.Matrix([[0,1],
                       [1,0]])
xcoupler = sp.Matrix([[coeff,coeff],
                  [coeff*sp.exp(sp.I*3/2*sp.pi),coeff*sp.exp(sp.I*1/2*sp.pi)]])



def mas2rad(x):
    ''' Convenient little function to convert milliarcsec to radians '''
    return(x * 4.8481368110953599e-09) # = x*np.pi/(180*3600*1000)

def rad2mas(x):
    '''  convert radians to mas'''
    return(x / 4.8481368110953599e-09) # = x / (np.pi/(180*3600*1000))

def sp2np(mat):
    try:
        npmat = np.array(sp.N(S), dtype=np.complex128)
    except:
        npmat = np.array(sp.N(S).tolist(), dtype=np.complex128)
        print("Had to convert to a list")
    return npmat


def printlatex(symbolic, imaginary_unit="j"):
    prefix = "\\begin{equation}\n  "
    suffix = "\n\end{equation}"
    print(prefix + sp.latex(symbolic, imaginary_unit=imaginary_unit) + suffix, file=sys.stdout)

def pol2cart(rho, theta):
    x = -np.imag(rho*np.exp(1j*theta*np.pi/180))
    y = np.real(rho*np.exp(1j*theta*np.pi/180))
    return np.array([x, y])

def get_contrast(self, signal, cov):
    nbkp = thekpo.kpi.nbkp
    xsi = chi2.ppf(1.-pfa, nbkp)
    c0 = 1.e3 #Fixing reference contrast
    
def expected_numbers(Na):
    print("Assuming nonredundant baselines")
    Nn = Na -1
    Nbl = np.math.factorial(Na)/\
        (np.math.factorial(2)*np.math.factorial(Na-2))
    print(Nbl, "baselines")
    Nderivatives = Nn + np.math.factorial(Nn)/\
                (np.math.factorial(2)*np.math.factorial(Nn-2))
    print(Nderivatives, "second order derivatives")
    Nnulls = np.math.factorial(Nn)
    print(Nnulls, "Number of nulls")
    Nthk = Nbl - (Na - 1)
    print(Nthk, "robust observables")
    print(Nthk*2, "independant nulls")


class kernuller(object):

    def __init__(self, tel_array, wl):
        """
        Creation of a kernel-nuller object:
        
        tel_array   : an array of (E,N) coordinates
                      for the stations
        wl          : the central wavelength for calculations
        """
        self.varying_instrumental_errors = False
        self.shotnoise = False
        self.pups = tel_array
        self.phi0 = 0.
        self.Na = tel_array.shape[0]
        self.wl = wl
        self.phierr = 0.001 #rms phase error in the pupil (radians)
        self.amperr = 0.0
        self.crosstalk = np.identity(self.Na)
        self.amps = np.ones(self.Na)
        self.phi = np.zeros(self.Na)
        self.Np = None
        self.S = None
        self.A = None
        self.K = None
        self.paper_difforder = True
        self.Taylor_order = 2
        self.perms = None
    def summary_properties(self, verbose=False, latex=False):
        if self.Np is not None:
            print("Shape of Np:",self.Np.shape)
        else:
            print("No Np matrix defined")
        if self.S is not None:
            print("Shape of S:",self.S.shape)
        else:
            print("No S matrix defined")
        if self.A is not None:
            print("Shape of A:",self.A.shape)
        else:
            print("No A matrix defined")
        try:
            self.As
            print("Shape of As:",self.As.shape)
            if verbose:
                sp.pprint(As)
            if latex:
                printlatex(As)
        except NameError:
            print("No symbolic value of A found")
        except:
            print("Error while examining the symbolic matrix")
            
    def build_classical_model(self,theta=np.pi/2, verbose=False):
        """
        Builds a kernel nuller purely based on the the matrices N and S
        hard-coded from Martinache2018. Only K is built by SVD.
        
        theta    : The value of the parameter theta (does not affect A or K)
        verbose  : Whether the calculations are detailed in the console

        Warning: The A matrix seem to suffer from an inversion between 
        penultimate and ultimate column. We lef it as found in the paper.

        """
        if self.Np is not None:
            print("The model will be replaced")
        else:
            print("Building a model from scratch")
            
        if self.Na != 4:
            print("Error, this construction method is dedicated to 4 aperture arrays")
            return
        
        N =  np.array([[1, 1, 1, 1],
                     [1, 1, -1, -1],
                     [1, -1, 1, -1],
                     [1, -1, -1, 1]]) / np.sqrt(4)

        if verbose:
            print("The N matrix")
            print(N)
        Np = N[1:,:]
        if verbose:
            print("The Np matrix")
            print(Np)
        S = np.array([[1,               np.exp(1j*theta),            0],
                    [-np.exp(-1j*theta), 1,                           0],
                    [1,                  0,            np.exp(1j*theta)],
                    [-np.exp(-1j*theta), 0,                           1],
                    [0,                  1,            np.exp(1j*theta)],
                    [0,                  -np.exp(-1j*theta),            1]]) / np.sqrt(4)
        if verbose:
            print("The S matrix")
            print(S)
        A = np.array([[1, 1, 1, -1, 0, 0],
                     [1, 1, 1, -1, 0, 0],
                     [1, 1, 1, 0, 0, -1],
                     [1, 1, 1, 0, 0, -1],
                     [1, 1, 1, 0, -1, 0],
                     [1, 1, 1, 0, -1, 0]]) * 1/4
        if verbose:
            print("The A matrix")
            print(A)
            
        u,s,vh = np.linalg.svd(A.T)
        take = np.concatenate((s<1e-6,np.ones(u.shape[1]-s.shape[0], dtype=np.bool)))
        #print(take)
        K = vh[take]
        if verbose:
            print("The K matrix")
            print(K)
            
        self.Np = Np
        self.S = S
        self.A = A    
        self.K = K
        
        return
    
    def build_symbolic_model(self,theta=sp.pi/2, verbose=False):
        """
        Builds a kernel nuller based on the paper nuller matrices, but using
        symbolic differentiation to compute the A matrix.
        verbose  : Whether the calculations are detailed in the console

        Warning: For now, the matrices are not normalized!!

        The algorithm uses hard-coded N and S matrices from Martinache2018

        if self.paper_difforder is set to False, the listing order of 
        the differential terms (for calculating A) is arbitrary based
        on the itertools.combinations_with_replacements function
        """
        if self.Np is not None:
            print("The model will be replaced")
        else:
            print("Building a model from scratch")
            
        if self.Na != 4:
            print("Error, this construction method is dedicated to 4 aperture arrays")
            return
        Ns = sp.Matrix([[1, 1, 1, 1],
                        [1, 1, -1, -1],
                        [1, -1, 1, -1],
                        [1, -1, -1, 1]]) / sp.sqrt(4)
        Nps = Ns[1:,:]
        if verbose:
            print("The Ns matrix")
            sp.pprint(Ns)
            printlatex(Ns)
        if verbose:
            print("The Nps matrix")
            sp.pprint(Nps)
            printlatex(Nps)
        thetas = sp.symbols("theta")
        Ss = sp.Matrix([[1,                 sp.exp(sp.I*thetas),            0],
                        [-sp.exp(-sp.I*thetas),1,                           0],
                        [1,                 0,            sp.exp(sp.I*thetas)],
                        [-sp.exp(-sp.I*thetas),0,                           1],
                        [0,                 1,            sp.exp(sp.I*thetas)],
                        [0,               -sp.exp(-sp.I*thetas),            1]]) / sp.sqrt(4)
        Ms = Ss @ Nps
        if verbose:
            print("The Ss matrix")
            sp.pprint(Ss)
            printlatex(Ss)
            print("The Ms matrix")
            sp.pprint(Ms)
            printlatex(Ms)
            
        thephis = sp.MatrixSymbol("phi", self.Na, 1)
        thephis = sp.Matrix(sp.symbols('phi1:{}'.format(self.Na+1), real=True))
        phis = sp.Matrix([thephis[i] for i in range(thephis.shape[0])])
        phis[0] = 0
        Ek = sp.Matrix([ sp.exp(-sp.I * phis[i]) for i in range(phis.shape[0]) ])
        Ekp = sp.Matrix([ 1 -sp.I * phis[i] for i in range(phis.shape[0])])
        #Here, it is a pain to do elementwise
        Eouts = Ms @ Ekp
        Iouts = sp.Matrix([(sp.Abs(Eouts[i]))**2 for i in range(Ms.shape[0])])
        Ioutssubs = Iouts.subs(thetas, theta)
        if verbose:
            print("Eouts")
            print(Eouts)
            printlatex(Eouts)
            print("Iouts")
            #sp.pprint(Iouts)
            #printlatex(Eouts)
            print("Ioutssubs")
            sp.pprint(Ioutssubs)
            printlatex(Ioutssubs)
        self.Eouts = Eouts
        self.Iouts = Iouts
        self.Ioutssubs = Ioutssubs
        
        cols = []
        if self.paper_difforder:
            square = np.array([np.arange(1,self.Na), np.arange(1,self.Na)]).T
            cross = np.array(list(itertools.combinations(range(1,self.Na), 2)))
            diffs = np.vstack((square, cross))
        else :
            diffs = np.array(list(itertools.combinations_with_replacement(np.arange(1,self.Na), 2)))
        self.diffs = diffs
        As = sp.Matrix([Ioutssubs.diff(phis[diffs[i,0]]).diff(phis[diffs[i,1]]) for i in range(diffs.shape[0])])
        As = As.reshape(Ioutssubs.shape[0], Ioutssubs.shape[0]).T

        A = np.array(As.tolist()).astype(np.float64)
        if verbose:
            print("The As matrix")
            sp.pprint(As)
            print(As)
            printlatex(As)
            print("The A matrix")
            print(A)   
        u,s,vh = np.linalg.svd(A.T)
        take = np.concatenate((s<1e-6,np.ones(vh.shape[1]-s.shape[0], dtype=np.bool)))
        print(take)
        K = vh[take]
        if verbose:
            print("The K matrix")
            print(K)
            
        self.Nps = Nps
        self.Np = np.array(self.Nps.subs(thetas,theta)).astype(np.complex128)
        self.Ss = Ss
        self.S = np.array(self.Ss.subs(thetas,theta)).astype(np.complex128)
        self.A = A
        self.K = K
        
        return
        
        
    def incoming_light(self, amp, binary=None):
        """
        Uses station positions to generates incoming light from the sky.
        amp      :    The amplitude of the main star flux by telescope
        binary   :    The binary parameters (standard kernel format (rho,theta,c)
                        if None the source is a point-source.
        returns an array of complex electric fields
        """
        E = amp * np.ones(self.Na, dtype=np.complex128)
        if binary is not None:
            sepcart = pol2cart(binary[0], binary[1])
            sepcartrad = mas2rad(sepcart)#The separation of the compag in radians
            piston_shifts = self.pups.dot(sepcartrad) 
            phi = 2 * np.pi * piston_shifts / self.wl
            comp = 1 / binary[2] * np.exp(1j*phi)
            E = np.vstack((E,comp))
        return E
    
    def update_instrumental_errors(self):
        """
        This function updates the instrumental errors.
        Currently this uses random uncorrelated errors.
        """
        self.phi = np.random.normal(size=self.Na, scale=self.phierr)
        self.amps = np.random.normal(size=self.Na, scale=self.amperr, loc=1.)
        
    def apply_instrumental_effects(self, E):
        """
        E       :   An input array of complex electric fields of shape self.Na
        Updates the errors IFF self.varying_instrumental_errors is True
        Applies the instrumental errors defined by: 
        
        self.amps : the amplitude values for each inputs
        self.phi  : the phase errors for each inputs
        
        returns the array of affected complex electric fields
        """
        if self.varying_instrumental_errors:
            self.update_instrumental_errors()
        phasor = self.amps * np.exp(1j*self.phi) #here, it is a complex pĥasor
        return phasor * E
    
    def instrument_propagation(self, E):
        """
        Simulates the propagation of light into the instument:
        
        E         : The complex input electric field to propagate
        
        self.shotnoise  : controls whether the output is affected by shot noise
        self.crosstalk  : a matrix that can be used to apply crosstalk on the inputs
        eturns an array of output intensities.
        """
        E1 = self.crosstalk.dot(E.T).T
        E2 = self.Np.dot(E1.T).T
        E3 = self.S.dot(E2.T).T
        Io = np.sum(np.abs(E3)**2, axis=0)
        if self.shotnoise:
            Io = np.random.poisson(Io).astype(np.float64)
        return Io
    def get_I(self, binary=None, amp=1):
        """
        Encapsulates the generation of signal and its propagation
        
        binary   :    The binary parameters (standard kernel format (rho,theta,c)
        amp      :    The amplitude of the main star flux by telescope
        
        Returns an array of output intensities.
        
        """
        E = self.incoming_light(amp, binary=binary)
        E_affected = self.apply_instrumental_effects(E)
        I0 = self.instrument_propagation(E_affected)
        return I0
    def test_target(self, binary=None, n=1000, amp=1):
        """
        Shortcut to generate a bunch of realizations of the output intensities
        
        binary   :    The binary parameters (standard kernel format (rho,theta,c)
        amp      :    The amplitude of the main star flux by telescope
        """
        outIs = np.array([self.get_I(binary=binary, amp=amp) for i in range(n)])
        return outIs
    def copy(self):
        return copy.deepcopy(self)
    

    def set_kept_nulls(self, kept_nulls, verbose=False):
        """
        Crop the matrices to discard redundancy

        kept_nulls  : an array of the nuller output to keep
        verbose     : Whether the function shows the result
        """
        if self.Np.shape[0]<kept_nulls.shape[0]:
            print("Error: more nulls requested than available")
            return


        Nps = sp.Matrix([self.Nps[i,:] for i in kept_nulls])
        self.perms = np.array([self.perms[i,:] for i in kept_nulls])
        Ms = sp.Matrix((sp.Matrix([np.ones(Nps.shape[1],dtype=np.int)]),Nps))
        if verbose:
            print("The Ms matrix")
            sp.pprint(Ms)
            printlatex(Ms)

        Ss = sp.Matrix(np.identity(Nps.shape[0]))

        thephis = sp.MatrixSymbol("phi", self.Na, 1)
        thephis = sp.Matrix(sp.symbols('phi1:{}'.format(self.Na+1), real=True))
        phis = sp.Matrix([thephis[i] for i in range(thephis.shape[0])])
        phis[0] = 0
        Ek = sp.Matrix([ sp.exp(-sp.I * phis[i]) for i in range(phis.shape[0]) ])
        #Ekp = sp.Matrix([ 1 -sp.I * phis[i] for i in range(phis.shape[0])])
        Ekp = sp.Matrix([sp.series(sp.exp(-sp.I*phis[i]),x=phis[i], x0=0, n=self.Taylor_order).removeO() for i in range(self.Na)])
        #Here, it is a pain to do elementwise
        Eouts = Nps @ Ekp
        Iouts = sp.Matrix([(sp.Abs(Eouts[i]))**2 for i in range(Nps.shape[0])])
        Ioutssubs = Iouts#.subs(thetas, theta)
        if verbose:
            print("Eouts")
            print(Eouts)
            printlatex(Eouts)
            print("Iouts")
            #sp.pprint(Iouts)
            #printlatex(Eouts)
            print("Ioutssubs")
            sp.pprint(Ioutssubs)
            printlatex(Ioutssubs)
        self.phis = phis
        self.Eouts = Eouts
        self.Iouts = Iouts
        self.Ioutssubs = Ioutssubs

        cols = []
        if self.paper_difforder:
            square = np.array([np.arange(1,self.Na), np.arange(1,self.Na)]).T
            cross = np.array(list(itertools.combinations(range(1,self.Na), 2)))
            diffs = np.vstack((square, cross))
        else :
            diffs = np.array(list(itertools.combinations_with_replacement(np.arange(1,self.Na), 2)))
        self.diffs = diffs
        As = sp.Matrix([Ioutssubs.diff(phis[diffs[i,0]]).diff(phis[diffs[i,1]]) for i in range(diffs.shape[0])])
        As = As.reshape(diffs.shape[0], Ioutssubs.shape[0]).T
        self.As = As
        A = sp.re(self.As)
        A = np.array(A.tolist()).astype(np.float64)
        if verbose:
            sp.pprint("The As matrix")
            print(As)
            print("The A matrix")
            print(A)   
        u,s,vh = np.linalg.svd(A.T)
        take = np.concatenate((s<1e-6,np.ones(vh.shape[1]-s.shape[0], dtype=np.bool)))
        K = vh[take]
        if verbose:
            print("The K matrix")
            print(K)

        self.Nps = Nps
        self.Ms = Ms
        self.Np = np.array(self.Nps).astype(np.complex128)
        self.Ss = Ss
        self.S = np.array(self.Ss).astype(np.complex128)
        self.A = A
        self.K = K

        return

    def build_procedural_model(self, verbose=False, onlyNp=False, kept=None):
        """
        Builds a kernel nuller through procedural algorithm.
        verbose  : Whether the calculations are detailed in the console

        Warning: For now, the matrices are not normalized!!
        The algorithm mostly requires the number of apertures self.Na 

        if self.paper_difforder is set to False, the listing order of 
        the differential terms (for calculating A) is arbitrary based
        on the itertools.combinations_with_replacements function
        """
        if self.Np is not None:
            print("The model will be replaced")
        else:
            print("Building a model from scratch")

        if self.Na >= 9:
            print("Warning, this makes huge matrices")
            #return
        #self.Nthk = len(list(itertools.combinations(np.arange(self.Na), 2))) - (self.Na - 1)

        Nbl = np.math.factorial(self.Na)/\
            (np.math.factorial(2)*np.math.factorial(self.Na-2))
        self.Nthk = Nbl - (self.Na - 1)
        thetas = sp.symbols("theta")

        thethetas = sp.Matrix([k * (2 * sp.pi / self.Na) for k in range(self.Na)], real=True)
        amp = 1/sp.sqrt(self.Na)
        perms = np.array(list(itertools.permutations(np.arange(1,self.Na,1, dtype=np.int32))))
        perms = np.hstack((np.zeros((perms.shape[0],1), dtype=np.int64), perms))
        if kept is not None:
            perms = perms[kept]
        #Adding zero phasors for the bright output in row 1
        perms = np.vstack((np.zeros((1,perms.shape[1]), dtype=np.int64), perms))
        self.perms = perms
        phasors = []
        for i in range(self.perms.shape[0]):
            phasors.append([amp*sp.exp(sp.I * thethetas[self.perms[i,j]]) for j in range(self.perms.shape[1])])
        phasors = sp.Matrix(phasors)
        #Ns = sp.Matrix([sp.ones(1,phasors.shape[1]),phasors])
        Ms = phasors
        #Warning: here we don't use a mask: assuming line 1 is the bright
        Nps = Ms[1:,:]#sp.Matrix((sp.Matrix([np.ones(Nps.shape[1],dtype=np.int)]),Nps))
        if verbose:
            print("The Ms matrix")
            sp.pprint(Ms)
            printlatex(Ms)

        #Ss = sp.Matrix(np.identity(Nps.shape[0]))
        if onlyNp:
            print("Stopping after building Np")
            self.Nps = Nps
            self.Np = np.array(self.Nps).astype(np.complex128)
            #self.Ss = Ss
            self.S = np.identity(self.Np.shape[0]).astype(np.complex128)
            return

        thephis = sp.MatrixSymbol("phi", self.Na, 1)
        thephis = sp.Matrix(sp.symbols('phi1:{}'.format(self.Na+1), real=True))
        phis = sp.Matrix([thephis[i] for i in range(thephis.shape[0])])
        phis[0] = 0
        Ek = sp.Matrix([ sp.exp(-sp.I * phis[i]) for i in range(phis.shape[0]) ])
        #Ekp = sp.Matrix([ 1 -sp.I * phis[i] for i in range(phis.shape[0])])
        Ekp = sp.Matrix([sp.series(sp.exp(-sp.I*phis[i]),x=phis[i], x0=0, n=self.Taylor_order).removeO() for i in range(self.Na)])
        #Here, it is a pain to do elementwise
        Eouts = Nps @ Ekp
        Iouts = sp.Matrix([(sp.Abs(Eouts[i]))**2 for i in range(Nps.shape[0])])
        Ioutssubs = Iouts#.subs(thetas, theta)
        if verbose:
            print("Eouts")
            print(Eouts)
            printlatex(Eouts)
            print("Iouts")
            #sp.pprint(Iouts)
            #printlatex(Eouts)
            print("Ioutssubs")
            sp.pprint(Ioutssubs)
            printlatex(Ioutssubs)
        self.phis = phis
        self.Eouts = Eouts
        self.Iouts = Iouts
        self.Ioutssubs = Ioutssubs

        cols = []
        if self.paper_difforder:
            square = np.array([np.arange(1,self.Na), np.arange(1,self.Na)]).T
            cross = np.array(list(itertools.combinations(range(1,self.Na), 2)))
            diffs = np.vstack((square, cross))
        else :
            diffs = np.array(list(itertools.combinations_with_replacement(np.arange(1,self.Na), 2)))
        self.diffs = diffs
        As = sp.Matrix([Ioutssubs.diff(phis[diffs[i,0]]).diff(phis[diffs[i,1]]) for i in range(diffs.shape[0])])
        As = As.reshape(diffs.shape[0], Ioutssubs.shape[0]).T
        self.As = As
        A = sp.re(self.As)
        A = np.array(A.tolist()).astype(np.float64)
        if verbose:
            sp.pprint("The As matrix")
            print(As)
            print("The A matrix")
            print(A)   
        u,s,vh = np.linalg.svd(A.T)
        take = np.concatenate((s<1e-6,np.ones(vh.shape[1]-s.shape[0], dtype=np.bool)))
        K = vh[take]
        if verbose:
            print("The K matrix")
            print(K)

        self.Nps = Nps
        self.Ms = Ms
        self.Np = np.array(self.Nps).astype(np.complex128)
        #self.Ss = Ss
        self.S = np.identity(self.Np.shape[0]).astype(np.complex128)
        self.A = A
        self.K = K

        return
    def build_generative_model(self,perms, verbose=False, onlyNp=False, mask=None):
        """
        Builds a kernel nuller based on a matrix of permutations of phasors.
        
        perms    : The matrix corresonding to phase indices for M (n'th
                    phasor on the circle)
        verbose  : Whether the calculations are detailed in the console

        Warning: For now, the matrices are not normalized!!
        The algorithm mostly requires the number of apertures self.Na 

        if self.paper_difforder is set to False, the listing order of 
        the differential terms (for calculating A) is arbitrary based
        on the itertools.combinations_with_replacements function
        """
        if self.Np is not None:
            print("The model will be replaced")
        else:
            print("Building a model from scratch")

        if self.Na >= 9:
            print("Warning, this makes huge matrices")
            #return
        #self.Nthk = len(list(itertools.combinations(np.arange(self.Na), 2))) - (self.Na - 1)

        Nbl = np.math.factorial(self.Na)/\
            (np.math.factorial(2)*np.math.factorial(self.Na-2))
        self.Nthk = Nbl - (self.Na - 1)
        thetas = sp.symbols("theta")

        thethetas = sp.Matrix([k * (2 * sp.pi / self.Na) for k in range(self.Na)], real=True)
        self.perms = perms
        amp = 1/sp.sqrt(self.Na)
        phasors = []
        print(thethetas)
        for i in range(self.perms.shape[0]):
            phasors.append([amp*sp.exp(sp.I * thethetas[self.perms[i,j]]) for j in range(self.perms.shape[1])])
        phasors = sp.Matrix(phasors)
        #Ns = sp.Matrix([sp.ones(1,phasors.shape[1]),phasors])
        Ms = phasors
        
        if mask is None:
            mask = np.ones(Ms.shape[0],dtype=np.bool)
            mask[0] = False
            
        Nps = []
        for i in range(self.perms.shape[0]):
            if mask[i]:
                Nps.append([amp*sp.exp(sp.I * thethetas[self.perms[i,j]]) for j in range(self.perms.shape[1])])
        Nps = sp.Matrix(Nps)
        
        if verbose:
            print("The Ms matrix")
            sp.pprint(Ms)
            printlatex(Ms)

        #Ss = sp.Matrix(np.identity(Nps.shape[0]))
        if onlyNp:
            print("Stopping after building Np")
            self.Nps = Nps
            self.Np = np.array(self.Nps).astype(np.complex128)
            #self.Ss = Ss
            self.S = np.identity(self.Np.shape[0]).astype(np.complex128)
            return

        thephis = sp.MatrixSymbol("phi", self.Na, 1)
        thephis = sp.Matrix(sp.symbols('phi1:{}'.format(self.Na+1), real=True))
        phis = sp.Matrix([thephis[i] for i in range(thephis.shape[0])])
        phis[0] = 0
        Ek = sp.Matrix([ sp.exp(-sp.I * phis[i]) for i in range(phis.shape[0]) ])
        #Ekp = sp.Matrix([ 1 -sp.I * phis[i] for i in range(phis.shape[0])])
        Ekp = sp.Matrix([sp.series(sp.exp(-sp.I*phis[i]),x=phis[i], x0=0, n=self.Taylor_order).removeO() for i in range(self.Na)])
        #Here, it is a pain to do elementwise
        Eouts = Nps @ Ekp
        Iouts = sp.Matrix([(sp.Abs(Eouts[i]))**2 for i in range(Nps.shape[0])])
        Ioutssubs = Iouts#.subs(thetas, theta)
        if verbose:
            print("Eouts")
            print(Eouts)
            printlatex(Eouts)
            print("Iouts")
            #sp.pprint(Iouts)
            #printlatex(Eouts)
            print("Ioutssubs")
            sp.pprint(Ioutssubs)
            printlatex(Ioutssubs)
        self.phis = phis
        self.Eouts = Eouts
        self.Iouts = Iouts
        self.Ioutssubs = Ioutssubs

        cols = []
        if self.paper_difforder:
            square = np.array([np.arange(1,self.Na), np.arange(1,self.Na)]).T
            cross = np.array(list(itertools.combinations(range(1,self.Na), 2)))
            diffs = np.vstack((square, cross))
        else :
            diffs = np.array(list(itertools.combinations_with_replacement(np.arange(1,self.Na), 2)))
        self.diffs = diffs
        As = sp.Matrix([Ioutssubs.diff(phis[diffs[i,0]]).diff(phis[diffs[i,1]]) for i in range(diffs.shape[0])])
        As = As.reshape(diffs.shape[0], Ioutssubs.shape[0]).T
        self.As = As
        A = sp.re(self.As)
        A = np.array(A.tolist()).astype(np.float64)
        if verbose:
            sp.pprint("The As matrix")
            print(As)
            print("The A matrix")
            print(A)   
        u,s,vh = np.linalg.svd(A.T)
        take = np.concatenate((s<1e-6,np.ones(vh.shape[1]-s.shape[0], dtype=np.bool)))
        K = vh[take]
        if verbose:
            print(take)
            print("The K matrix")
            print(K)

        self.Nps = Nps
        self.Ms = Ms
        self.Np = np.array(self.Nps).astype(np.complex128)
        #self.Ss = Ss
        self.S = np.identity(self.Np.shape[0]).astype(np.complex128)
        self.A = A
        self.K = K

        return
    
    def build_model_from_matrix(self,M, verbose=False, onlyNp=False, kept=None, mask=None):
        """
        Builds a kernel nuller through procedural algorithm.
        verbose  : Whether the calculations are detailed in the console

        Warning: For now, the matrices are not normalized!!
        The algorithm mostly requires the number of apertures self.Na 

        if self.paper_difforder is set to False, the listing order of 
        the differential terms (for calculating A) is arbitrary based
        on the itertools.combinations_with_replacements function
        """
        if self.Np is not None:
            print("The model will be replaced")
        else:
            print("Building a model from scratch")

        #self.Nthk = len(list(itertools.combinations(np.arange(self.Na), 2))) - (self.Na - 1)
        
        #Testing if input matrix is sympy:

        
        if isinstance(M, sp.Basic):
            Ms = M
        else :
            Ms = sp.Matrix(M)
        if mask is None:
            mask = np.ones(Ms.shape[0],dtype=np.bool)
            mask[0] = False
            
        Nps = []
        for i in range(Ms.shape[0]):
            if mask[i]:
                Nps.append(Ms[i,:])
        Nps = sp.Matrix(Nps)
        
            

        Nbl = np.math.factorial(self.Na)/\
            (np.math.factorial(2)*np.math.factorial(self.Na-2))
        self.Nthk = Nbl - (self.Na - 1)
        thetas = sp.symbols("theta")

        thethetas = sp.Matrix([k * (2 * sp.pi / self.Na) for k in range(self.Na)], real=True)

        if verbose:
            print("The Ms matrix")
            sp.pprint(Ms)
            printlatex(Ms)

        #Ss = sp.Matrix(np.identity(Nps.shape[0]))
        if onlyNp:
            print("Stopping after building Np")
            self.Nps = Nps
            self.Np = np.array(self.Nps).astype(np.complex128)
            #self.Ss = Ss
            self.S = np.identity(self.Np.shape[0]).astype(np.complex128)
            return

        thephis = sp.MatrixSymbol("phi", self.Na, 1)
        thephis = sp.Matrix(sp.symbols('phi1:{}'.format(self.Na+1), real=True))
        phis = sp.Matrix([thephis[i] for i in range(thephis.shape[0])])
        phis[0] = 0
        Ek = sp.Matrix([ sp.exp(-sp.I * phis[i]) for i in range(phis.shape[0]) ])
        #Ekp = sp.Matrix([ 1 -sp.I * phis[i] for i in range(phis.shape[0])])
        Ekp = sp.Matrix([sp.series(sp.exp(-sp.I*phis[i]),x=phis[i], x0=0, n=self.Taylor_order).removeO() for i in range(self.Na)])
        #Here, it is a pain to do elementwise
        Eouts = Nps @ Ekp
        Iouts = sp.Matrix([(sp.Abs(Eouts[i]))**2 for i in range(Nps.shape[0])])
        Ioutssubs = Iouts#.subs(thetas, theta)
        if verbose:
            print("Eouts")
            print(Eouts)
            printlatex(Eouts)
            print("Iouts")
            #sp.pprint(Iouts)
            #printlatex(Eouts)
            print("Ioutssubs")
            sp.pprint(Ioutssubs)
            printlatex(Ioutssubs)
        self.phis = phis
        self.Eouts = Eouts
        self.Iouts = Iouts
        self.Ioutssubs = Ioutssubs

        cols = []
        if self.paper_difforder:
            square = np.array([np.arange(1,self.Na), np.arange(1,self.Na)]).T
            cross = np.array(list(itertools.combinations(range(1,self.Na), 2)))
            diffs = np.vstack((square, cross))
        else :
            diffs = np.array(list(itertools.combinations_with_replacement(np.arange(1,self.Na), 2)))
        self.diffs = diffs
        As = sp.Matrix([Ioutssubs.diff(phis[diffs[i,0]]).diff(phis[diffs[i,1]]) for i in range(diffs.shape[0])])
        As = As.reshape(diffs.shape[0], Ioutssubs.shape[0]).T
        self.As = As
        A = sp.re(self.As)
        A = np.array(A.tolist()).astype(np.float64)
        if verbose:
            sp.pprint("The As matrix")
            print(As)
            print("The A matrix")
            print(A)   
        u,s,vh = np.linalg.svd(A.T)
        take = np.concatenate((s<1e-6,np.ones(vh.shape[1]-s.shape[0], dtype=np.bool)))
        K = vh[take]
        if verbose:
            print(take)
            print("The K matrix")
            print(K)

        self.Nps = Nps
        self.Ms = Ms
        self.Np = np.array(self.Nps).astype(np.complex128)
        #self.Ss = Ss
        self.S = np.identity(self.Np.shape[0]).astype(np.complex128)
        self.A = A
        self.K = K

        return
    def get_rank(self, params, thr=1e-6, verbose=False, mode="kernels"):
        """
        Returns the rank of the nuller,
        that is the number of independant kenelises outputs
        
        params    :   A bunch of random binary parameters in the ROI 
                    (must be more than the original number of nulled outputs)
        thr       :   The threshold to consider the singular values to be 0
        """
        outamps = np.array([self.get_I(binary=params[i]) for i in range(params.shape[0])])
        if mode=="kernels":
            outkers = self.K.dot(outamps.T).T
            u, s, vh = np.linalg.svd(outkers)
            order = np.count_nonzero(s>=thr)
        else :
            u, s, vh = np.linalg.svd(outamps)
            order = np.count_nonzero(s>=thr)
        if verbose:
            print(s)
        return order

    def legacy_nulled_output(self,matrix=None, valtheta=sp.pi/2, simplified=False):
        """
        Plots the complex combinations of the nulled outputs of 
        legacy-style (S.N) nuller.
        valtheta    : the value of the parameter theta for the nuller
        simplified  : whether to compute the sum of each telescope's
                        phased contribution.
        """
        if matrix is None:
            M = self.Ss@(self.Nps)
        else :
            M = matrix
        nx = M.shape[0]
        fig, axs = plt.subplots(nx,1, sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(6,8), )
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                theterm = M[i,j]
                theargs = np.array([theterm.args[i].evalf(subs={"theta":valtheta}) for i in range(len(theterm.args))], dtype=np.complex128)
                os = np.array(theterm.evalf(subs={"theta":valtheta}),dtype=np.complex128)
                if simplified:
                    axs[i].plot([0, np.real(os)], [0, np.imag(os)], color="C"+str(j), linewidth=5)
                    axs[i].text(np.real(os/2), np.imag(os/2), str(j))
                else:
                    for k in range(len(theargs)):
                        axs[i].plot([0+np.real(os/10), np.real(theargs[k]) + np.real(os/10)],
                                    [0+np.imag(os/10), np.imag(theargs[k]) + np.imag(os/10)],
                                    color="C"+str(j), linewidth=5)
                        axs[i].text(np.real(os/2), np.imag(os/2), str(j))
            axs[i].set_aspect("equal")
        #fig.show()
        if simplified :
            fig.suptitle("Simplified representation\n of nulled outputs")
        else :
            fig.suptitle("Unsimplified representation\n of nulled outputs")
        return fig, axs
    
    def plot_response_maps(self,data, nx=1,ny=None,plotsize=2, cmap="coolwarm", extent=None,
                          title="On-sky respnse maps", plotspaces=(0.2,0.2),
                          unit="mas"):
        """
        Plots a set of maps into a single figure
        data    : A set of maps piled along the 0 axis
        """
        if ny is None:
            ny = data.shape[0]//nx
        if data.shape[0]%nx != 0:
            ny = ny +1
        ntot = data.shape[0]
        
        gmin, gmax = np.nanmin(data), np.nanmax(data)
        
        fig, axs = plt.subplots(ny,nx,sharex='col', sharey='row',
                                gridspec_kw={'hspace': plotspaces[0], 'wspace': plotspaces[1]}, figsize=(plotsize*nx,plotsize*ny +0.1), )
        for i in range(axs.shape[0]):
            for j in range(nx):
                if nx==1:

                    #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
                    im = axs[i].imshow(data[i,:,:], vmin=gmin, vmax=gmax, cmap=cmap, extent=extent)
                    axs[i].set_aspect("equal")
                else:
                    if i*j<=data.shape[0]:
                        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
                        im = axs[i,j].imshow(data[i*nx+j,:,:], vmin=gmin, vmax=gmax, cmap=cmap, extent=extent)
                        axs[i,j].set_aspect("equal")

        #if nx==1:
        #    fig.colorbar(im, ax=axs[-1])
        #else:
        #    fig.colorbar(im, ax=axs[-1,-1])
        if title:
            plt.suptitle(title)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("On-sky RA position (%s)"%(unit))
        plt.ylabel("On-sky DEC position (%s)"%(unit))
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        return fig, axs
    
    def plot_nulled_outputs(self,matrix=None, nx=4, verbose=False):
        """
        Plots the complex combinations of the nulled outputs for 
        procedural kernullers.
        nx       : The number of columns of plots
        """
        special = True
        if matrix is None:
            matrix = self.Np
            special = False
        if verbose:
            if perms is not None:
                for perm in self.perms:
                    print(perm)
        if matrix.shape[0]%nx != 0:
            nx = nx +1
        ny = matrix.shape[0]//nx
        if self.Na == None:
            nx = 2
            ny = 1
            print("Plotting exception for 3T")
            fig, axs = plt.subplots(ny,nx, sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(1.5*nx,1.5*ny+1), )
            for i in range(axs.shape[0]):
                for k in range(matrix.shape[1]):
                    #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
                    axs[i].plot([0,matrix[i,k].real], [0,matrix[i,k].imag],
                                          color="C"+str(k), linewidth=5)
                    axs[i].set_aspect("equal")
                    axs[i].text(matrix[i,k].real+0.1, matrix[i,k].imag, str(k))
        
        else :
            fig, axs = plt.subplots(ny,nx,
                                    gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(1.5*nx,1.5*ny+1), )
            for i in range(axs.shape[0]):
                for j in range(nx):
                    for k in range(matrix.shape[1]):
                        if nx==1:
                            
                            #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
                            axs[i].plot([0,matrix[i*nx+j,k].real], [0,matrix[i*nx+j,k].imag],
                                          color="C"+str(k), linewidth=5)
                            axs[i].set_aspect("equal")
                            axs[i].text(matrix[i*nx+j,k].real*0.8, matrix[i*nx+j,k].imag*0.8, str(k))
                        else:
                            
                            #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
                            axs[i,j].plot([0,matrix[i*nx+j,k].real], [0,matrix[i*nx+j,k].imag],
                                          color="C"+str(k), linewidth=5)
                            axs[i,j].set_aspect("equal")
                            axs[i,j].text(matrix[i*nx+j,k].real*0.8, matrix[i*nx+j,k].imag*0.8, str(k))
        if special:
            fig.suptitle("The null configurations\n of all the %d outputs"%(matrix.shape[0]))
        else :
            fig.suptitle("The null configurations\n of the %d nulled outputs"%(matrix.shape[0]))
        #fig.tight_layout()
        plt.show()
        return fig, axs
    
    
    def plot_outputs_smart(self,matrix=None, inputfield=None, nx=4,ny=None,
                       verbose=False, osfrac=0.1, plotsize=2, plotspaces=(0.5,0.5), onlyonelegend=True,
                       labels=True, legend=True,legendsize=8, title=None, projection="polar",
                       out_label=None, rmax=None, show=True):
        """
        Plots the complex combinations of the nulled outputs for 
        procedural kernullers.
        nx       : The number of columns of plots
        osfrac   : The fraction of the amplitude of phasors to use as offset.
        """
        special = True
        
        
        if matrix is None:
            matrix = self.Np
            special = False
            
        if inputfield is not None:
            initialmatrix = matrix
            matrix = matrix.dot(np.diag(inputfield))
            outvec = initialmatrix.dot(inputfield)
        else :
            initialmatrix = np.zeros((0,0))
            outvec = None
        if verbose:
            if perms is not None:
                for perm in self.perms:
                    print(perm)
        if ny is None:
            ny = matrix.shape[0]//nx
        if matrix.shape[0]%nx != 0:
            ny = ny +1
        ntot = matrix.shape[0]
        
        if projection=="polar":
            sharex="all"
            sharey="all"
            text_coords="polar"
        else:
            sharex="all"
            sharey="all"
            text_coords="data"
        if rmax is None:    
            rmax=np.max(matrix)
        
        fig, axs = plt.subplots(ny,nx,sharex=sharex, sharey=sharey,
                                gridspec_kw={'hspace': plotspaces[0], 'wspace': plotspaces[1]}, figsize=(plotsize*nx,plotsize*matrix.shape[0]//nx+0.5),
                                subplot_kw=dict(projection=projection))
        
        for i in range(axs.shape[0]):
            for j in range(nx):
                plotted = []
                adjust=[]
                for k in range(matrix.shape[1]):
                    if (i*nx+j)<matrix.shape[0]:
                        item = matrix[i*nx+j,k]
                        baseoffset = 0
                        if item==0:
                            continue
                        
                        plotted = plotitem(axs, item, plotted, nx, i, j, k,
                                           osfrac=osfrac, baseoffset=baseoffset,
                                           linestyle="-", label=str(k), labels=labels,
                                           projection=projection, rmax=rmax)
                
                plotted2 = []
                adjus2t = []
                for k in range(initialmatrix.shape[1]):
                    if (i*nx+j)<initialmatrix.shape[0]:
                        item = initialmatrix[i*nx+j,k]
                        baseoffset = 0
                        if item==0:
                            continue
                        
                        plotted = plotitem(axs, item, plotted2, nx, i, j, k,
                                           osfrac=osfrac, baseoffset=baseoffset,
                                           linestyle="--", label=None, labels=False,
                                           projection=projection, rmax=rmax, linewidth=2)
                if (outvec is not None) and ((i*nx+j)<matrix.shape[0]):
                    if nx==1:
                        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
                        axs[i].plot([np.angle(outvec[i]),np.angle(outvec[i])], [0,np.abs(outvec[i])],
                                      color="k", linewidth=5,
                                      linestyle="-", label=label)
                        axs[i].set_aspect("equal")
                        axs[i].set_ylim(0,rmax)
                        thetas = np.linspace(0,2*np.pi,100)
                        axs[i].plot(thetas, np.ones_like(thetas)*np.abs(outvec[i]),
                                      color="k")

                    else:

                        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
                        axs[i,j].plot([0,np.angle(outvec[i*nx+j])], [0,np.abs(outvec[i*nx+j])],
                                      color="k", linewidth=5,
                                      linestyle="-", label="output")
                        axs[i,j].set_aspect("equal")
                        axs[i,j].set_ylim(0,rmax)
                        thetas = np.linspace(0,2*np.pi,100)
                        axs[i,j].plot(thetas, np.ones_like(thetas)*np.abs(outvec[i*nx+j]),
                                      color="k")
                
                    
                if legend:
                    if nx==1:
                        if onlyonelegend:
                            if i==0:
                                axs[i].legend(prop={'size': legendsize})
                        else :
                            axs[i].legend(prop={'size': legendsize})
                    else :
                        if onlyonelegend:
                            if i*nx+j ==0:
                                axs[i,j].legend(prop={'size': legendsize})
                        else:
                            axs[i,j].legend(prop={'size': legendsize})
                            
                if out_label is not None:
                    if nx==1:
                        axs[i,j].text(90., rmax, str(out_label[i]), size=15,
                                ha="right", va="top", 
                                bbox=dict(boxstyle="square",
                                #facecolor="none",
                                alpha=0.2,
                                facecolor="C"+str((i)//2),
                                edgecolor="C"+str((i)//2),
                                ))
                    else :
                        axs[i,j].text(np.pi/2, rmax, str(out_label[i*nx+j]), size=15,
                                ha="right", va="top", 
                                bbox=dict(boxstyle="square",
                                #facecolor="none",
                                alpha=0.2,
                                facecolor="C"+str((i*nx+j)//2),
                                edgecolor="C"+str((i*nx+j)//2),
                                ))
        #eliminating the empty plots
        rowstoremove = np.prod(matrix, axis=1) == 0
        xT = np.arange(0, 2*np.pi,np.pi/4)
        xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
                r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
        for i in np.flip(np.arange(matrix.shape[0])):
            
            fig.axes[i].set_xticks(xT)
            fig.axes[i].set_xticklabels(xL)
            if rowstoremove[i] :
                fig.axes[i].remove()
            
                            
        if title is not False:
            if title is None:
                title = "The null configurations\n of all the %d outputs"%(matrix.shape[0])
            fig.suptitle(title)
        #fig.tight_layout()
        if show:
            plt.show()
        return fig, axs
    

    
    
    def plot_pupils(self, offset=5, s=20):
        a = plt.figure()
        for i in range(self.pups.shape[0]):
            plt.scatter(self.pups[i,0], self.pups[i,1], marker="o", s=s)
            plt.text(self.pups[i,0]+offset, self.pups[i,1]+offset,i)
        plt.gca().set_aspect("equal")
        plt.xlabel("East position (m)")
        plt.ylabel("North position (m)")
        plt.title("A map of the interferometer")
        plt.show()
        return a

    def find_valid_nulls(self, params, nit=100):
        thnnull = len(list(itertools.combinations(np.arange(self.Na),2))) - self.Na +1
        for i in range(nit):
            randid = (self.Np.shape[0] * np.random.rand(thnnull*2)).astype(np.int16)
            akernuller = self.copy()
            akernuller.set_kept_nulls(randid)
            nks = akernuller.get_rank(params)
            print(nks)
            if nks >= akernuller.Nthk:
                break
            elif i == nit-1:
                print("could not find a good one")
                return None
        randid.sort()
        return randid
    
    
    def find_valid_nulls_systematic(self, params, target_rank=None):
        """
        Looks for 
        """
        if target_rank is None:
            target_rank = self.Nthk
        indices = np.arange(self.Np.shape[0])
        print("all indices", indices)
        for k in np.arange(self.Np.shape[0]-1, 0,-1):
            print("trying to delete %d"%(k))
            for i in range(indices.shape[0]-1,0,-1):
                randid = np.delete(indices, i)
                akernuller = self.copy()
                akernuller.set_kept_nulls(randid)
                nks = akernuller.get_rank(params, thr=1e-8)

                if nks >= target_rank:
                    indices = np.delete(indices,i)
                    print("We deleted %d"%(i))
                    break
                else:
                    print("could not delete",i, nks)
            if akernuller.K.shape[0] == akernuller.Nthk:
                print("We deleted enough of them!")
                break
        return indices
    
def generative_random_pruning(matrix, niter):
    ss = []
    matrices = []
    recipes = []
    for i in tqdm(range(niter)):
        na = matrix.shape[1]
        runningmat = matrix[1:,:].copy()
        running_approx = np.array(sp.N(runningmat), dtype=np.complex128)
        combinermat = sp.zeros(0, na)
        combiner_approx = np.zeros((0,matrix.shape[1]))

        it = 0
        nbad = 0
        picks = []
        while combinermat.shape[0] < (na-2)*(na-1)  :
            pick = np.random.randint(0, runningmat.shape[0])
            picks.append(pick)
            #Warning: non idempotent code below!! (because of row_del)
            arow = runningmat[pick,:]
            arow_approx = running_approx[pick,:]
            #deleting the element
            runningmat.row_del(pick)
            running_approx = np.delete(running_approx, (pick), axis=0)

            complementpick = isin_fast(sp.conjugate(arow), running_approx)

            acomplement = runningmat[complementpick, :]
            acomplement_approx = running_approx[complementpick, :]
            #deleting the complementary element
            runningmat.row_del(complementpick)
            running_approx = np.delete(running_approx, (complementpick), axis=0)

            testmat = combinermat.copy()
            test_approx = combiner_approx.copy()

            testmat = testmat.row_insert(testmat.shape[0], arow)
            testmat = testmat.row_insert(testmat.shape[0], acomplement)
            test_approx = np.vstack((test_approx, arow_approx, acomplement_approx))
            signature = get_signature_fast(test_approx)
            #print(test_approx)
            #print(signature)
            it += 1
            if np.count_nonzero(signature[1:,1:]>(na-2)) > 0:
                #print("Not balanced!!")
                nbad+=1
            else :
                #print("Seems ok")
                combinermat = testmat
                combiner_approx = test_approx
                last_good_signature = signature
        picks = np.array(picks)
        combinermat = combinermat * 1/sp.sqrt(na-2)
        combinermat = combinermat.row_insert(0,matrix[0,:])
        Mn = np.array(sp.N(combinermat), dtype=np.complex128)
        u, s, vh = np.linalg.svd(Mn)
        ss.append(s)
        if np.linalg.norm(s-np.ones_like(s)) <= 1e-6 :
            print("FOUND ONE!!!")
            matrices.append(combinermat)
            #print(combinermat)
            print(picks)
            recipes.append(picks)
    ss = np.array(ss)
    return matrices, recipes

def generative_from_recipe(matrix, recipe):
    na = matrix.shape[1]
    runningmat = matrix[1:,:].copy()
    running_approx = np.array(sp.N(runningmat), dtype=np.complex128)
    combinermat = sp.zeros(0, na)
    combiner_approx = np.zeros((0,6))

    it = 0
    nbad = 0
    picks = []
    for pick in recipes[0]  :
        #pick = np.random.randint(0, runningmat.shape[0])
        #Warning: non idempotent code below!! (because of row_del)
        arow = runningmat[pick,:]
        arow_approx = running_approx[pick,:]
        #deleting the element
        runningmat.row_del(pick)
        running_approx = np.delete(running_approx, (pick), axis=0)

        complementpick = isin_fast(sp.conjugate(arow), running_approx)

        acomplement = runningmat[complementpick, :]
        acomplement_approx = running_approx[complementpick, :]
        #deleting the complementary element
        runningmat.row_del(complementpick)
        running_approx = np.delete(running_approx, (complementpick), axis=0)

        testmat = combinermat.copy()
        test_approx = combiner_approx.copy()

        testmat = testmat.row_insert(testmat.shape[0], arow)
        testmat = testmat.row_insert(testmat.shape[0], acomplement)
        test_approx = np.vstack((test_approx, arow_approx, acomplement_approx))
        signature = get_signature_fast(test_approx)
        #print(test_approx)
        #print(signature)
        it += 1
        if np.count_nonzero(signature[1:,1:]>(na-2)) > 0:
            #print("Not balanced!!")
            nbad+=1
        else :
            #print("Seems ok")
            combinermat = testmat
            combiner_approx = test_approx
            last_good_signature = signature
            picks.append(pick)
    combinermat = combinermat * 1/sp.sqrt(na-2)
    combinermat = combinermat.row_insert(0,matrix[0,:])
    Mn = np.array(sp.N(combinermat), dtype=np.complex128)
    u, s, vh = np.linalg.svd(Mn)
    print(s)

### Some utility functions to run the generative random approach
def isin_fast(vector, approxmat, threshold=1e-8):
    approxvec = np.squeeze(np.array(sp.N(vector), dtype=np.complex128))
    #print(approxvec)
    mask = np.linalg.norm(approxmat-approxvec[None,:], axis=1) < threshold
    loc = np.squeeze(np.argwhere(mask))
    #print(mask)
    return loc
def get_signature_fast(matrix, threshold=1e-8):
    """
    Warning: by default, this function will take the first row of phasors
    to be the list of phasors to count.
    """
    
    refphasors = matrix[0,:]
    signature = np.array([np.count_nonzero(np.abs(matrix - refphasors[i])<threshold, axis=0) for i in range(refphasors.shape[0])])
    return signature


def pairwise_kernel(n):
    """
    Returns a kernel for pairwise nullers
    n :  The number of nulls
    """
    kervec = np.zeros(n)
    kervec[:2] = np.array([1,-1])
    customker = np.array([np.roll(kervec,2*i) for i in range(kervec.shape[0]//2)])
    return(customker)

VLTI = np.array([[-9.925,-20.335],
                    [14.887,30.502],
                    [44.915,66.183],
                    [103.306,44.999]])
mykernuller = kernuller(VLTI[:3],3.6e-6)
mykernuller.build_procedural_model(verbose=False)
tricoupler = sp.Matrix([mykernuller.Ms[1,:], mykernuller.Ms[0,:], mykernuller.Ms[2,:]])

def plotitem(axs, item, plotted, nx, i, j,k, osfrac=0.1,
             baseoffset=0, linestyle="-", label="X", linewidth=5,
             labels=True, projection="polar", rmax=1.):
    
    offset = osfrac*np.abs(item)*np.exp(1j*(np.angle(item)+np.pi/2))
    while item in plotted:
        item += offset
        baseoffset += offset
    if projection=="polar":
        a0=np.angle(baseoffset)
        b0=np.abs(baseoffset)
        a1=np.angle(item)
        b1=np.abs(item)
        a2=np.angle(offset)
        b2=np.abs(offset)
    else:
        a0=np.real(baseoffset)
        b0=np.imag(baseoffset)
        a1=np.real(item)
        b1=np.imag(item)
        a2=np.real(offset)
        b2=np.imag(offset)
    if nx==1:
        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
        axs[i].plot([a0,a1], [b0,b1],
                      color="C"+str(k), linewidth=linewidth,
                      linestyle=linestyle, label=label)
        if labels:
            axs[i].text(0.95*a1, 0.9*b1, str(k))
        axs[i].set_aspect("equal")
        axs[i].set_ylim(0,rmax)

    else:

        #axs[i,j].scatter(matrix[i*nx+j,k].real, matrix[i*nx+j,k].imag)
        axs[i,j].plot([a0,a1], [b0,b1],
                      color="C"+str(k), linewidth=linewidth,
                      linestyle=linestyle, label=label)
        if labels:
            axs[i,j].text(0.95*a1, 0.9*b1, str(k))
        axs[i,j].set_aspect("equal")
        axs[i,j].set_ylim(0,rmax)
    plotted.append(item)
    return plotted