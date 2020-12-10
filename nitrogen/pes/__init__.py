"""
nitrogen.pes
------------

Potential energy surface utilities and library.

"""

import importlib 


def loadpes(pesname):
    """
    Load built-in PES.

    Parameters
    ----------
    pesname : str
        PES name.

    Returns
    -------
    pes : DFun
        The PES as a DFun object.

    """
    
    try : 
        mod = importlib.import_module("nitrogen.pes.library." + pesname)
    except ModuleNotFoundError:
        raise ValueError("PES name is not recognized.")
        
        
    return mod.PES