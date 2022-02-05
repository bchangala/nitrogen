"""
cfour.py

CFOUR interface
"""

import nitrogen.dfun as dfun
import nitrogen.constants 
import numpy as np

import os 
import shutil 
import re 

__all__ = ['CFOUR']

class CFOUR(dfun.DFun):
    """
    
    A simple CFOUR interface for accessing single point
    energies and derivatives.
    
    Attributes
    ----------
    params : dict
        The CFOUR keyword values used for the calculation.
    natoms : int
        The number of atoms. This equals `nx`/3.
    atomic_symbols : list
        The atomic symbols of each atom.
    work_dir : str
        The work directory.
    cleanup : bool
        The work directory clean-up flag.
    
    """
    
    def __init__(self, atomic_symbols, params, 
                 work_dir = './scratch',
                 cleanup = True, units = 'angstrom'):
        
        """
        Create a new CFOUR interface.
        
        Parameters
        ----------
        atomic_symbols : list
            The atomic symbols of each atom.
        params : dict
            CFOUR keyword and value pairs.
        work_dir : str, optional
            The path to the work directory. The default is
            \'.\\scratch\'.
        cleanup : bool, optional
            If True (default), work directories will be
            deleted after use. 
        units : {'angstrom', 'bohr'}, optional
            The Cartesian units.
            
            
        Notes
        -----
        
        All elements in `params` will be added as keywords to the
        ``*CFOUR()`` section of a CFOUR ``ZMAT`` input file. The
        ``COORD``,``UNITS``, ``DERIV_LEV``, and ``PRINT``
        keywords are handled automatically.
        These should not be supplied by the user.
        
        """
        
        
        natoms = len(atomic_symbols) # number of atoms
        nx = 3*natoms # number of coordinates 
        
        # Initialize the DFun object
        #
        # We hard-code the maxderiv to 2,
        # which is all CFOUR will except 
        # via the DERIV_LEVEL keyword
        #
        super().__init__(self._f_cfour, nf = 1, nx = nx,
                         maxderiv = 2, zlevel = None)
        
        
        # Make sure Cartesian coordinates are on
        # and that the units are Angstroms
        #
        params = params.copy() # Create a new copy 
        params["COORD"] = "CARTESIAN"
        if units == 'angstrom':
            params["UNITS"] = "ANGSTROM"  
        elif units == 'bohr':
            params["UNITS"] = "BOHR"
        else:
            raise ValueError('Unexpected units value')
        
        self.params = params 
        self.natoms = natoms 
        self.atomic_symbols = atomic_symbols 
        self.work_dir = work_dir 
        self.cleanup = cleanup
        
    
    def _f_cfour(self, X, deriv = 0, out = None, var = None):
        """
        Evaluate CFOUR energy and derivatives
        """
        
        if var is None:
            var = [i for i in range(self.nx)]
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        
        # X has shape (nx,) + base_shape 
        base_shape = X.shape[1:]
        if out is None:
            out = np.ndarray((nd, self.nf) + base_shape, dtype = np.float64)
        
        
        # Loop over each input geometry in
        # serial fashion
        Xflat = np.reshape(X, (self.nx, -1))
        npts = Xflat.shape[1] # The number of jobs 
        
        out_flat = np.zeros((nd, self.nf, npts))
        
        for i in range(npts):
            
            jobstr = f'job{i:05d}'
            jobdir = os.path.join(self.work_dir, jobstr)
            
            #
            # Create the work space
            try:
                os.makedirs(jobdir)
            except FileExistsError:
                # Remove previous job directory
                shutil.rmtree(jobdir)
                os.makedirs(jobdir)
            
            
            # Energy calculation only
            
            #
            # Create the ZMAT text file
            # for a single-point energy
            #
            with open(os.path.join(jobdir, 'ZMAT'), 'w') as file:
                file.write('nitrogen-cfour-interface\n')
                # 
                # Write Cartesian coordinates
                for j in range(self.natoms):
                    file.write(self.atomic_symbols[j] + " ")
                    for k in range(3):
                        xval = Xflat[3*j+k, i] # Atom j, coordinate k = x,y,z
                        file.write(f'{xval:.15f} ')
                    
                    file.write("\n")
                file.write("\n")
                
                # Write options
                file.write("*CFOUR(")
                
                first = True
                for keyword, value in self.params.items():
                    if not first:
                        file.write("\n") # new line
                    else:
                        first = False 
                    file.write(keyword + "=" + value)
                    
                # Write the deriviative level keyword
                file.write("\nDERIV_LEV=") 
                if deriv == 0:
                    file.write("ZERO")
                elif deriv == 1:
                    file.write("ONE")
                elif deriv == 2:
                    file.write("TWO")
                else:
                    raise ValueError("Unexpected deriv level")
                
                # Write the PRINT keyword
                file.write("\nPRINT=")
                if deriv == 0:
                    file.write("0")
                else:
                    file.write("1")
                
                file.write(")\n")
                
            # Save current word dir 
            current_wd = os.getcwd()
            # Now change to the job directory
            os.chdir(jobdir)
            # Execute cfour
            os.system('xcfour > out')
            # and go back
            os.chdir(current_wd) 
            
            #
            # Find the energy in the string
            #
            #   "The final electronic energy is     XXXXXXXXXXXXXXXXX a.u."
            #
            found = False
            with open(os.path.join(jobdir, 'out'), 'r') as file:
                for line in file:
                    if re.search('final electronic energy', line):
                        energy = line.split()[5] # The sixth field is the energy, a.u.
                        found = True
            
            if not found:
                raise RuntimeError("CFOUR appears to have a problem.")
                
            out_flat[0,0,i] = energy

            #
            # Remove job directory
            #
            if self.cleanup:
                shutil.rmtree(jobdir)
                
            
        # Reshape output data to correct base_shape
        # and copy to out buffer
        np.copyto(out, out_flat.reshape((nd, self.nf) + base_shape))
        
        #
        # Finally, convert energy units
        # from Hartree to cm^-1
        #
        out *= nitrogen.constants.Eh 
        
        return out
        
    def __repr__(self):
        return f"CFOUR({self.atomic_symbols!r}, {self.params!r})"
        
        
