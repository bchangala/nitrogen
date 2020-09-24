"""
nitrogen.dvr
------------

This module provides support for discrete-variable 
representation (DVR) basis functions. The main object
is the :class:`DVR` class.

"""

import numpy as np



class DVR:
    """
    A 1D DVR basis class
    
    Attributes
    ----------
    start : float
        The DVR grid starting value.
    stop : float
        The DVR grid stopping value.
    num : int
        The number of DVR grid points.
    basis : {'sinc'}
        The DVR basis type.
    grid : ndarray
        The DVR grid points.
    D : ndarray
        First derivative operator in DVR representation.
    D2 : ndarray
        Second derivative operator in DVR representation.
    
    """
    
    def __init__(self, start = 0, stop = 1, num = 10, basis = 'sinc'):
        """
        Create a DVR object.

        Parameters
        ----------
        start : float, optional
            The DVR grid starting value. The default is 0.
        stop : float, optional
            The DVR grid stopping value. The default is 1.
        num : int, optional
            The number of DVR grid points. The default is 10.
        basis : {'sinc'}, optional
            The DVR basis type. The default is 'sinc'.

        """
        
        if stop <= start:
            raise ValueError("stop value must be larger than start value")
        if num < 2 :
            raise ValueError("num must be >= 2")
        
        self.start = start 
        self.stop = stop 
        self.num = num 
        self.basis = basis.lower()
        
        if self.basis == 'sinc' : 
            self.grid, self.D, self.D2 = _sincDVR(start, stop, num)
            self.wfs = lambda q : _sincDVRwfs(q, self.start, self.stop, self.num)
        else:
            raise ValueError("basis type '{}' not recognized".format(basis))
    
    
    def wfs(self, q):
        """
        Evaluate DVR basis wavefunctions.

        Parameters
        ----------
        q : ndarray
            A 1-D array of coordinate values.

        Returns
        -------
        wfs : ndarray
            An array of shape (`q`.size, :attr:`num`) containing
            the values of the DVR wavefunctions evaluated at 
            coordinate values `q`.

        """
        pass # This will be replaced by an instance method for a given DVR basis type
            
def _sincDVR(start,stop,num):
    """ 
    Calculate sinc DVR grid and operators
    
    Parameters
    ----------
    start : float
        Grid start value
    stop : float
        Grid stop value. Must be larger than start
    num : int
        Number of grid points. Must be >= 2.

    Returns
    -------
    grid : ndarray
        Array of DVR grid points
    D : ndarray
        First derivative operator in DVR basis, shape (num, num)
    D2 : ndarray (num,num)
        Second derivative operator in DVR basis, shape (num, num)

    """
    
    if stop <= start:
        raise ValueError("stop value must be larger than start value")
    if num < 2 :
        raise ValueError("num must be >= 2")
        

    grid = np.linspace(start,stop,num)
    
    D = np.zeros((num,num))
    D2 = np.zeros((num,num))

    delta = grid[1]-grid[0] # Grid spacing
    
    for i in range(num):
        for j in range(num):
            
            # Calculate first derivative operator D
            if i==j:
                pass #zero
            else:
                D[i,j] = (-1.0)**(i-j) / (delta * (i-j))
            
            # Calculate second derivative operator D2
            if i==j:
                D2[i,j] = -np.pi**2 / (3.0 * delta**2)
            else:
                D2[i,j] = -2.0 * (-1.0)**(i-j) / (delta**2 * (i-j)**2)
    
    
    return grid,D,D2

def _sincDVRwfs(q,start,stop,num):
    """
    Calculate sinc DVR wavefunctions.

    Parameters
    ----------
    q : ndarray
        A 1D array of coordinate values.
    start : float
        Sinc DVR grid start value.
    stop : float
        Sinc DVR grid stop value.
    num : int
        The number of DVR grid points

    Returns
    -------
    wfs : ndarray
        A (`q`.size, `num`) shaped array with
        the DVR wavefunctions evaluated at grid points `q`.

    """
    
    if np.ndim(q) != 1:
        raise ValueError("q must be 1-dimensional")
        
    nq = q.size
    
    wfs = np.ndarray((nq,num), dtype = q.dtype)
    
    grid = np.linspace(start,stop,num)
    delta = grid[1] - grid[0]
    
    for i in range(num):
        # Compute sinc DVR basis function at
        # grid position i
        # Note: np.sinc function is defined as sin(pi*x)/pi*x 
        #
        fi = np.sqrt(1.0/delta) * np.sinc( (q - grid[i])/delta )
        wfs[:,i] = fi
        
    return wfs
