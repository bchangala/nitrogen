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
    
    mod = importlib.import_module("nitrogen.pes.library." + pesname)
    
    return mod.PES