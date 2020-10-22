"""
nitrogen.constants

Physical constants and reference data.

"""

from .ame2016 import masses 



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
                    val.append(masses[item][0])
                except KeyError:
                    raise ValueError(f"There is no mass entry for {item:s}")
            else:
                val.append(item) # assume a literal float
            
    else: # Not a list, assume a single label
        if isinstance(label, str):
            try:
                val = masses[label][0]
            except KeyError:
                    raise ValueError(f"There is no mass entry for {label:s}")
        else:
            val = label # assume a literal float
        
    return val
    