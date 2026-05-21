# -*- coding: utf-8 -*-
"""
Carbon empirical force field 

from P. M. Tailor, R. J. Wheatley, and N. A. Besley 
Carbon 113, 299 (2017)
doi: 10.1016/j.carbon.2016.11.059
    
"""

import nitrogen.autodiff.cyad as adc
import nitrogen as n2 
from .cmm import pes 

###################
# Create a DFun for the Cartesian PES
#
def make_cmm_pes(n, input_fun = None):
    """
    Construct a DFun PES object.

    Parameters
    ----------
    n : integer
        The number of carbon atoms.
    

    Returns
    -------
    DFun
        The Cartesian PES in 3*n coordinates (Angstroms).

    """
    PESX = adc.ForwardDFun(pes, 1, 3*n, input_fun = input_fun) # The Cartesian function
    return PESX 
#
###################
