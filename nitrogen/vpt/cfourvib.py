"""
nitrogen.vpt.cfourvib
---------------------

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
        freq_str = lines[pos].strip()
        if freq_str[-1] == 'i':
            # imaginary frequency 
            freq_i = float(freq_str[:-1]) * 1j 
        else:
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
    
def sample_QUADRATURE(filename, num_sample, use_bohr = False,
                      rng_seed = None, stddev = 1.0):
    """
    Generate sample points using a CFOUR QUADRATURE file.

    Parameters
    ----------
    filename : str
        The QUADRATURE file path.
    num_sample : integer
        The number of sample points.
    use_bohr : bool, optional
        If True, return Cartesian coordintes in bohrs.
        The default is False.
    rng_seed : integer, optional
        The RNG seed. 
    stddev : float, optional
        The standard deviation of the sampled dimensionless normal coordinates, :math:`q`.

    Returns
    -------
    X : (3*N, num_sample) ndarray
        The sampled geometries
        
    Notes
    -----
    
    The harmonic oscillator ground-state wavefunction in terms of :math:`q` is
    :math:`\\psi(q) \\propto \\exp[-q^2/2]`, and the ground-state density is
    :math:`\\vert \\psi \\vert ^2 \\propto \\exp[-q^2]`. These distributions
    have standard deviations of :math:`1` and 
    :math:`\\sqrt{\\frac{1}{2}} \\approx 0.707`, respectively. Therefore,
    setting the option `stddev` = 0.707 will sample the harmonic ground-state
    density exactly. The default option (`stddev` = 1.0) samples the ground-state
    amplitude, which is :math:`\\sqrt{2}`` wider.
    

    """

    # Read the QUADRATURE file
    freq, T, ref_geo = read_QUADRATURE(filename, use_bohr = use_bohr)
    
    nvib = T.shape[1] # The number of vibrational modes 
    
    # Generate (nvib, num_sample) random numbers 
    rng = np.random.default_rng(rng_seed)
    # The default standard-normal has a std_dev of 1.0
    x = rng.standard_normal((nvib, num_sample)) # <x**2> = 1.00 
    # Rescale to `stddev`
    q = stddev * x 
    
    # Calculate displaced geometries 
    
    X = T @ q + ref_geo.reshape((-1,1)) 
    
    return X 
    
def QUADRATURE2xyz(filename, elements, out = "out.xyz", comment = ""):
    """
    Generate a .xyz file from a CFOUR QUADRATURE file.

    Parameters
    ----------
    filename : str
        The QUADRATURE file path.
    elements : array_like
        The element labels, e.g. ['H','O','H']
    out : str, optional
        The output file. The default is "out.xyz".
    comment : str, optional
        The .xyz file comment string. The default is None.

    Returns
    -------
    None.
    
    Notes
    -----
    Note that the standard CFOUR QUADRATURE file uses bohr units,
    and the default .xyz file uses Angstroms units. This unit
    conversion is performed in this function.
    
    """
    
    freq, T, ref_geo = read_QUADRATURE(filename, use_bohr = False)
    
    N = T.shape[0] // 3  # the number of atoms 
    Nvib = T.shape[1] # the number of vibrations
    
    ref_geo = np.reshape(ref_geo, (N,3) )
    T = np.reshape(T,(N,3,Nvib))
    
    with open(out,"w") as file:
        
        # Write reference geometry
        file.write(f"{N:d}\n") # number of atoms
        file.write("Reference geometry " + comment + "\n")
        
        for j in range(N): # For each atom
            e = elements[j] # element label
            x = ref_geo[j,0] # x
            y = ref_geo[j,1] # y
            z = ref_geo[j,2] # z
            
            file.write(f"{e:s}   {x:.15f} {y:.15f} {z:.15f} \n")
        
        # Write each vibration 
        for i in range(Nvib):
            file.write(f"{N:d}\n") # number of atoms
            file.write(f"Vibration {i+1:d} ({freq[i]:.2f} cm^-1) " + comment + "\n")
            
            for j in range(N): # For each atom
                e = elements[j] # element label
                x = ref_geo[j,0] # x
                y = ref_geo[j,1] # y
                z = ref_geo[j,2] # z
                
                # Displacements
                dx = T[j,0,i]
                dy = T[j,1,i]
                dz = T[j,2,i]
                
                file.write(f"{e:s}   {x:.15f} {y:.15f} {z:.15f} {dx:.15f} {dy:.15f} {dz:.15f} \n")
    
    return 