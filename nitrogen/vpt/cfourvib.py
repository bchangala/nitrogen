"""
cfourvib.py

CFOUR vibrational file processing and
interface routines.

"""

import numpy as np 
import nitrogen.constants
import nitrogen.autodiff.forward as adf 


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

def read_cubic(filename, nvib = None, offset = 7):
    """
    Read a CFOUR cubic force constants text file
    
    Parameters
    ----------
    filename : str
        The file path
    nvib : int, optional
        The number of vibrational modes. If None, the total
        will be inferred from the input file.
    offset : int, optional
        The mode numbering offset from zero. The default is 7,
        which is usually what CFOUR files require (6 for 
        rot-trans modes and 1 for zero-indexing).
    
    Returns
    -------
    F : ndarray
        The scaled derivative including up to cubic derivatives.
        (The zeroth, first, and second derivatives are zeroed.)
    
    Notes
    -----
    The deriative array is returned in standard scaled format, i.e.
    the derivatives are divided by the factorial of their
    respective multi-index.
    
    """
    
    with open(filename,'r') as file:
        lines = file.readlines()

    # Figure out maximum index value 
    if nvib is None:
        maxindex = 0 
        for line in lines :
            indices = [int(x) for x in line.split()[0:3]]
            maxindex = max(maxindex, max(indices))
        nvib = maxindex - offset + 1 


    # Create a derivative array up to cubics 
    nd = adf.nderiv(3, nvib) 
    F = np.zeros((nd,)) 
    midx = np.zeros((nvib,), dtype = np.uint32) 

    # Calculate a binomial table 
    nck = adf.ncktab(nvib + 2, min(nvib, 2)) 

    for line in lines:
        
        if not line:
            continue # skip any empty lines 
            
        toks = line.split()
        if len(toks) != 4:
            raise ValueError('4 entries expected on each line!')            
        
        # Grab the indices and apply offset
        # Sort the modes in ascending order
        modes = [int(x) - offset for x in toks[0:3]]
        modes.sort() 
        
        # Determine the position in the derivative array
        # of this multi-index `midx`
        midx.fill(0) 
        for i in range(3):
            midx[modes[i]] += 1
        pos = adf.idxpos(midx, nck) 
        
        # Grab the force constant 
        coeff = float(toks[-1])
        
        # Determine what permutation factor needs to
        # be applied before storing the force
        # constant in the derivative array
        # (`modes` is in ascending sorted order)
        #
        if modes[0] == modes[1] and modes[1] == modes[2]:
            # iii type
            #
            F[pos] = coeff / 6.0  # 1/3!
        elif modes[0] == modes[1] or modes[1] == modes[2]:
            # iij or ijj type 
            F[pos] = coeff / 2.0  # 1/2! * 1/1!
        else: 
            # ijk type 
            F[pos] = coeff 
        
    # F now contains all elements 
    
    return F 
    
    