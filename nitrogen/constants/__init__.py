"""
nitrogen.constants
------------------

Physical constants and reference data.

NITROGEN generally uses an Å/u/cm\ :sup:`-1`
unit system, defined as follows

.. list-table::
   :widths: 20 20
   :header-rows: 1
   
   * - Dimension
     - Unit
   * - [length]
     - :math:`\\text{Å}`
   * - [mass]
     - :math:`\\text{u}`
   * - [energy]
     - :math:`hc\\times\\text{cm}^{-1}`
   * - [temperature]
     - :math:`K`
   * - [electric charge]
     - :math:`e`

All constants in the :mod:`nitrogen.constants` module
are reported in these units, which are convenient for
nuclear motion calculations. As of CODATA 2018, the units of length
and energy have exact relationships to their SI counterparts.
The unit of mass (u, unified atomic mass unit, i.e. one-twelth the
mass of a carbon atom), however, does not have an exact SI value. 
The atomic unit of mass (the electron mass) has a similar status.

===========================  =================================================================
Physical constants
==============================================================================================
``kb`` (``kB``)              Boltzmann constant :math:`k_B`
``h``                        Planck constant :math:`h`
``hbar``                     reduced Planck constant :math:`\hbar = h/(2\pi)`
``a0``                       Bohr radius :math:`a_0`
``c``                        speed of light
``me``                       electron mass
``NA``                       Avogadro constant :math:`N_A`
===========================  =================================================================



===========================  =================================================================
Unit conversions
==============================================================================================
``joule``                    joule, J
``kJ``                       kilojoule per mole, kJ/mol
``kcal``                     kilocalorie per mole, kcal/mol
``eV``                       electron-volt, eV
``Eh``                       hartree, :math:`E_h`
``hHz``                      energy equivalent of hertz, :math:`h \\times \\text{Hz}`
``second``                   second, s
``t_au``                     atomic unit of time, :math:`\hbar/E_h`
``debye``                    debye, D
===========================  =================================================================


=======================      =======================================
Look-up methods
====================================================================
:func:`constant_val`         Look up value by name.
:func:`constant_unc`         Look up uncertainty by name.
:func:`mass`                 Look up atomic mass by element symbol.
=======================      =======================================


==============================   =====================================
Version information
======================================================================
:func:`constants_version`        Version string for constants data.
:func:`mass_version`             Version string for atomic mass data.
==============================   =====================================

"""

from .ame2016 import _masses
from .codata2018 import _constants

#
# Common physical constants:
# All values are in [A,u,hc*cm^-1] unit system
#
kb      = _constants["kb"][0]       # Boltzman constant
kB      = kb                        # alias for `kb`
joule   = _constants["joule"][0]    # 1 Joule
h       = _constants["h"][0]        # Planck constant
hbar    = _constants["hbar"][0]     # Reduced Planck constant
a0      = _constants["a0"][0]       # Bohr radius
second  = _constants["second"][0]   # 1 second
eV      = _constants["eV"][0]       # 1 eV
Eh      = _constants["Eh"][0]       # 1 Hartree
c       = _constants["c"][0]        # Speed of light
me      = _constants["me"][0]       # Electron mass
debye   = _constants["debye"][0]    # 1 Debye
NA      = _constants["NA"][0]       # Avogadro constant
kJ      = _constants["kJ"][0]       # kJ/mol
kcal    = _constants["kcal"][0]     # kcal/mol
t_au    = _constants["t_au"][0]     # atomic unit of time (hbar/Eh)
hHz     = _constants["hHz"][0]      # h * Hz


#############################
# Define version functions
def mass_version():
    """ Version information for atomic mass data """
    return "ame2016"
def constants_version():
    """ Version information for physical constants data """
    return "codata2018"

#############################
# Define retrieval functions
#
def constant_val(name):
    """
    Retrieve a physical constant's value

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
    Retrieve a physical constant's uncertainty.
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
    Retrieve atomic mass data.

    Parameters
    ----------
    label : str or list of str
        Atomic label(s).

    Returns
    -------
    float or list of float
        The atomic mass in u.
        
    Notes
    -----
    Literal float values can be passed as `label`. They will be
    returned as is.

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
    