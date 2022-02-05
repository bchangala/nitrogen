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
        ``COORD``,``UNITS``, ``DERIV_LEV``, ``PRINT``, and ``VIB``
        keywords are handled automatically.
        These should not be supplied by the user. Only methods
        for which ``VIB=ANALYTIC`` (i.e. analytic Hessians) is
        available are supported. For other methods, use
        lower CFOUR derivatives with NITROGEN finite difference drivers.
        
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
        
        # Intermediate storage:
        # Flatten the base_shape to one dimension
        #
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
                    
                # Write the necessary keywords based
                # on deriv level
                #
                if deriv == 0:
                    file.write("\nDERIV_LEV=0") 
                    file.write("\nPRINT=0")
                elif deriv == 1:
                    file.write("\nDERIV_LEV=1") 
                    file.write("\nPRINT=1")
                elif deriv == 2:
                    file.write("\nDERIV_LEV=2") 
                    file.write("\nPRINT=1")
                    file.write("\nVIB=ANALYTIC")
                
                file.write(")\n")
                
            # Save current word dir 
            current_wd = os.getcwd()
            # Now change to the job directory
            os.chdir(jobdir)
            # Execute cfour
            os.system('xcfour > out')
            # and go back
            os.chdir(current_wd) 
            
            ######################
            # Parse CFOUR output
            ######################
            if deriv >= 0:
                #
                # Find the energy in the string
                #
                #   "The final electronic energy is     XXXXXXXXXXXXXXXXX a.u."
                #
                found = False
                with open(os.path.join(jobdir, 'out'), 'r') as file:
                    for line in file:
                        if re.search('final electronic energy', line):
                            energy = float(line.split()[5]) # The sixth field is the energy, a.u.
                            found = True
                
                if not found:
                    raise RuntimeError(f"Cannot find a CFOUR energy in {jobstr}")
                
                # Save energy
                # converting from hartree to cm**-1
                out_flat[0,0,i] = energy * nitrogen.constants.Eh
            
            if deriv >= 1:
                #
                # Find the gradient
                # 
                # This will be headed by 
                #       reordered gradient in QCOM coords for ZMAT order
                # followed by a line for each atom 
                # in the original ZMAT ordering we provided
                #
                # Note:
                #  "QCOMP" is the computation coordinates used by 
                #  CFOUR for most of its work
                #  "QCOM" (no "P") are the original coordinates passed
                #  in ZMAT translated to the COM frame
                #  
                #  Because energies and gradients are independent
                #  of total translations, we can use QCOM 
                #  derivatives.
                #
                found = False
                with open(os.path.join(jobdir, 'out'), 'r') as file:
                    for line in file:
                        if re.search('reordered gradient', line):
                            # found it
                            found = True
                            break 
                    if not found:
                        raise RuntimeError(f"Cannot find a CFOUR gradient in {jobstr}.")
                
                    # Now parse gradient lines
                    grad_all = np.zeros((self.nx,))
                    for j in range(self.natoms):
                        grad_str = file.readline().split() 
                        for k in range(3):
                            # Save the x, y, z components
                            grad_all[j*3 + k] = float(grad_str[k]) 
                    
                # Now copy the requested derivatives to
                # the output buffer, per the `var` order
                #
                # The CFOUR printed values are in units
                # of hartree/bohr. Convert this to 
                # cm**-1 / Angstrom
                #
                coeff = nitrogen.constants.Eh / nitrogen.constants.a0
                for k in range(len(var)):
                    # The derivative w.r.t. var[k]
                    out_flat[k+1,0,i] = grad_all[var[k]] * coeff 

            if deriv >= 2:
                #
                # Parse the Hessian data
                # 
                # VIB=ANALYTIC stores this in FCM
                #
                try:
                    fcm_raw = np.loadtxt(os.path.join(jobdir,'FCM'), 
                                         skiprows=1)
                except:
                    raise RuntimeError(f"Cannot find a CFOUR FCM file in {jobstr}.")
                if fcm_raw.shape != (3 * self.natoms**2 , 3):
                    raise RuntimeError(f"Unexpected FCM shape in {jobstr}.")
                
                fcm_1d = np.zeros((self.nx**2,))
                for j in range(3*self.natoms**2):
                    for k in range(3):
                        fcm_1d[j*3 + k] = fcm_raw[j,k]
                fcm_full = fcm_1d.reshape((self.nx, self.nx)) 
                
                #
                # Convert from Eh/a0**2 to cm**-1 / Angstrom**2
                fcm_full *= nitrogen.constants.Eh / (nitrogen.constants.a0**2) 
                
                # Copy to output buffer 
                idx = len(var) + 1 # The starting index for second derivatives
                for k1 in range(len(var)):
                    for k2 in range(k1, len(var)):
                        v1 = var[k1]
                        v2 = var[k2] 
                        # need the v1,v2 derivative
                        out_flat[idx, 0, i] = fcm_full[v1,v2] 
                        if v1 == v2:
                            out_flat[idx,0,i] *= 0.5 
                            # derivative array format includes
                            # 1/2! factor of diagonal second derivative
                        idx += 1 
                # done
                #########
                
                    
                    
            #
            # Remove job directory
            #
            if self.cleanup:
                shutil.rmtree(jobdir)
                
            
        # Reshape output data to correct base_shape
        # and copy to out buffer
        np.copyto(out, out_flat.reshape((nd, self.nf) + base_shape))
        
        return out
        
    def __repr__(self):
        return f"CFOUR({self.atomic_symbols!r}, {self.params!r})"
        
        
