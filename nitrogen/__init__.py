"""
NITROGEN
========

A Python package for quantum nuclear motion
calculations and rovibronic spectroscopy.

"""

# Import sub-packages and modules into namespace
from . import autodiff
from . import linalg 
from . import basis
from . import dfun 
from . import coordsys 
from . import ham
from . import constants
from . import pes
from . import tensor 
from . import scf 
from . import special
from . import math 
from . import vpt 

__version__ = '2.2.dev0'

import numpy as np 
import matplotlib.pyplot as plt 
import re 

#################################################################
# Define some top-level constants
pi = 3.14159265358979323846264338327950288419716939937510   # pi
deg = pi / 180.0 # 1 degree in radians
rad = 180.0 / pi # 1 radian in degrees 

def X2xyz(X, elements, filename = "out.xyz", comment = None):
    """
    Create an .xyz text file from an array of
    Cartesian positions.

    Parameters
    ----------
    X : (3*N,...) ndarray
        Cartesian position of N atoms.
    elements : list of str
        List of element identifiers
    filename : str, optional
        Output filename. The default is "out.xyz".
    comment : str, optional
        A comment string for the .xyz file. If None,
        a default will be used.

    Returns
    -------
    None.

    """
    
    if np.ndim(X) < 1 :
        raise ValueError("X must have at least 1 dimension.")
    if X.shape[0] % 3 != 0:
        raise ValueError("The first dimension of X must be a multiple of 3.")
    N = X.shape[0] // 3 # number of atoms 
    if N < 1:
        raise ValueError("The number of atoms must be >= 1")
    
    Xp = np.reshape(X, (N,3,-1)) 
    
    ng = Xp.shape[2] # The number of geometries 
    
    # Write .xyz file 
    with open(filename, 'w') as file :
        
        for i in range(ng): # For each geometry
            file.write(f"{N:d}\n") # number of atoms
            
            if comment is None: 
                file.write(f"Geometry {i:d}\n") # comment line
            else :
                file.write(f"{comment:s}\n") # supplied comment 
            
            for j in range(N): # For each atom
                e = elements[j] # element label
                x = Xp[j,0,i] # x
                y = Xp[j,1,i] # y
                z = Xp[j,2,i] # z
                
                file.write(f"{e:s}   {x:.15f} {y:.15f} {z:.15f} \n")
    # Done
    return 

def X2txt(X, filename = "grid.txt", write_index = True):
    """
    Create an indexed text file from an array of
    Cartesian positions.

    Parameters
    ----------
    X : (3*N,...) ndarray
        Cartesian position of N atoms.
    filename : str, optional
        Output filename. The default is "grid.txt".
    write_index : bool, optional
        Include the point index. The default is True.

    Returns
    -------
    None.

    """
    
    if np.ndim(X) < 1 :
        raise ValueError("X must have at least 1 dimension.")
    if X.shape[0] % 3 != 0:
        raise ValueError("The first dimension of X must be a multiple of 3.")
    N = X.shape[0] // 3 # number of atoms 
    if N < 1:
        raise ValueError("The number of atoms must be >= 1")
    
    Xp = np.reshape(X, (N,3,-1)) 
    
    ng = Xp.shape[2] # The number of geometries 
    
    # Write .txt file 
    with open(filename, 'w') as file :
        
        for i in range(ng): 
            if write_index:
                file.write(f"{i+1:5d} ") # write index 
            
            for j in range(N): # For each atom
                x = Xp[j,0,i] # x
                y = Xp[j,1,i] # y
                z = Xp[j,2,i] # z
                
                file.write(f"{x:-18.15f} {y:-18.15f} {z:-18.15f} ")
            
            file.write("\n")
    # Done
    return 

def podvr(prim_dvr, npo, i, qref, cs, pes_fun, masses, disp = 0):
    """
    Construct a potential-optimized contracted DVR based on the
    contrained body-fixed Hamiltonian.

    Parameters
    ----------
    prim_dvr : GenericDVR
        The primitive DVR.
    npo : int
        The number of contracted functions
    i : int
        The coordinate index to contract (w.r.t. `cs` and `qref` ordering).
    qref : list of float
        The reference geometry.
    cs : CoordSys
        The coordinate system.
    pes_fun : DFun or function
        The potential energy function
    masses : list of float
        The masses for the coordinate system.
    disp : integer, optional
        The print level. Default is 0. Level 1 prints eneriges.
        Level 2 plots wavefunctions

    Returns
    -------
    Contracted
        A contracted DVR object.

    """
        
    # Construct POs
    dvrs = [a for a in qref]
    dvrs[i] = prim_dvr 
    h1 = ham.hdpdvr_bfJ(dvrs, cs, pes_fun, masses, Jlist = 0)
    w,u = np.linalg.eigh(linalg.full(h1))
    
    if disp >= 1:
        print(f"The PODVR energies for coordinate {i:d} are")
        print(w[:npo])
        
    if disp >= 2:
        # plot wavefunctions
        scale = (w[1]-w[0]) / max(abs(u[:,0])) * 0.8 
        
        plt.figure() 
        for j in range(npo):
            plt.plot(prim_dvr.grid, scale * u[:,j] + w[j])
        plt.xlabel('q')
        plt.ylabel('Energy')

    return dvrs[i].contract(u[:,:npo])

def parseFormula(formula):
    """
    Parse a chemical formula into a list of element symbols.

    Parameters
    ----------
    formula : str
        The chemical formula

    Returns
    -------
    list
        The list of element symbols
        
    Examples
    --------
    >>> n2.parseFormula('H2O')
    ['H', 'H', 'O']
    >>> n2.parseFormula('CH3OH')
    ['C', 'H', 'H', 'H', 'O', 'H']
    >>> n2.parseFormula('CFBrClI')
    ['C', 'F', 'Br', 'Cl', 'I']
    
    """
    
    ###############
    # The formula is first split by capital letters,
    # then any numbers specify repetition.
    
    symbols = [] 
    
    toks = re.findall('[A-Z][^A-Z]*', formula)
    
    for t in toks:
        sym_num = re.split('(\d+)',t)
        
        sym = sym_num[0] 
        if len(sym_num) > 1 : 
            num = int(sym_num[1])
        else:
            num = 1 
            
        symbols = symbols + [sym]*num 
    
    return symbols 

    