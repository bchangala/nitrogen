"""
cfour_vib.py

CFOUR vibrational file processing and
interface routines.

"""

import numpy as np 
import nitrogen.constants

__all__ = ['read_QUADRATURE']



def read_QUADRATURE(filename, use_bohr = False):
    """
    Parse a CFOUR QUADRATURE file.

    Parameters
    ----------
    filename : str
        The QUADRATURE file path.
    use_bohr : bool, optional
        If True, return displacements and geometry in bohrs.
        The default is False.

    Returns
    -------
    freq : (nvib,) ndarray
        The harmonic frequencies
    T : (3*natom,nvib) ndarray
        The Cartesian displacement vectors, in Angstroms,
        of each dimensionless normal mode.
    ref_geo : (3*natom,) ndarray
        The reference Cartesian geometry in Angstroms.
    

    """

    # Read all lines of
    # the QUADRATURE file 
    with open(filename,'r') as file:
        lines = file.readlines()
    Nlines = len(lines)
    
    # Find the lines that contain
    # a '%'
    pct_lines = []
    for i in range(Nlines):
        if '%' in lines[i]:
            pct_lines.append(i+1) 
    
    # The number of atoms equals
    # the difference between the 
    # second and third occurences of '%'
    # minus 1.
    Natoms = pct_lines[2] - pct_lines[1] - 1
    Nvib = 3*Natoms - 6 # the number of vibrational modes
    
    # Collect the displacement vectors
    # and harmonic frequencies 
    T = np.zeros((3*Natoms, Nvib))
    freq = []
    
    pos = 1 
    for i in range(Nvib):
        freq_i = float(lines[pos].strip()) 
        pos += 2 
        for j in range(Natoms):
            disp_row = [float(x) for x in lines[pos].split() ]
            pos += 1
            # Store displacements in T
            for k in range(3): # x,y,z
                T[3*j + k, i] = disp_row[k]
        pos += 1 
        
        freq.append(freq_i)
    freq = np.array(freq) # Convert to ndarray
    
    
    # Now get the reference geometry
    ref_geo = np.zeros((3*Natoms,))
    for j in range(Natoms):
        ref_row = [float(x) for x in lines[pos].split() ]
        pos += 1
        for k in range(3): # x,y,z
            ref_geo[3*j + k] = ref_row[k]
            
    #
    # The displacements and reference geometry 
    # are assumed to be in bohr. Convert to 
    # Angstroms.
    # 
    
    if not use_bohr: 
        T *= nitrogen.constants.a0 
        ref_geo *= nitrogen.constants.a0
        
    return freq, T, ref_geo 