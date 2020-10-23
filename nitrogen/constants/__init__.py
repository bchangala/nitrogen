"""
nitrogen.constants

Physical constants and reference data.

"""

from .ame2016 import _masses
from .codata2018 import _constants

mass_version = "ame2016"
constants_version = "codata2018"



kb      = _constants["kb"][0]
joule   = _constants["joule"][0]
h       = _constants["h"][0]
hbar    = _constants["hbar"][0]
a0      = _constants["a0"][0]
second  = _constants["second"][0]
eV      = _constants["eV"][0]
Eh      = _constants["Eh"][0]
c       = _constants["c"][0]
me      = _constants["me"][0]
debye   = _constants["debye"][0]

#############################
# Define retrieval functions
#
def constant_val(name):
    """
    Retrieve value from constant dictionary

    Parameters
    ----------
    name : str
        The name of the constant

    Returns
    -------
    float
        The value

    """
    try:
        val = _constants[name][0]
    except KeyError:
        raise ValueError(f"There is no constant entry for {name:s}")
    
    return val

def constant_unc(name):
    """
    Retrieve a constant's uncertainty from constant dictionary.
    The uncertainty value does *not* reflect errors from 
    finite precision numerical arithmetic.

    Parameters
    ----------
    name : str
        The name of the constant

    Returns
    -------
    float
        The uncertainty

    """
    try:
        unc = _constants[name][1]
    except KeyError:
        raise ValueError(f"There is no constant entry for {name:s}")
    
    return unc

def mass(label):
    """
    Fetch atomic mass data (using AME2016)

    Parameters
    ----------
    label : str or list of str
        Atomic label(s).

    Returns
    -------
    float or list of float
        The atomic mass in u.

    """
    
    if isinstance(label, list):
        val = []
        for item in label:
            if isinstance(item, str):
                try:
                    val.append(_masses[item][0])
                except KeyError:
                    raise ValueError(f"There is no mass entry for {item:s}")
            else:
                val.append(item) # assume a literal float
            
    else: # Not a list, assume a single label
        if isinstance(label, str):
            try:
                val = _masses[label][0]
            except KeyError:
                    raise ValueError(f"There is no mass entry for {label:s}")
        else:
            val = label # assume a literal float
        
    return val
    