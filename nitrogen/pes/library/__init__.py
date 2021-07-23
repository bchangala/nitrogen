"""
nitrogen.pes.library
--------------------

A collection of potential energy surfaces. These should
be accessed via the :func:`nitrogen.pes.loadpes()` function.

All functions have Cartesian coordinate inputs (Angstroms) and 
return energies in cm**-1, unless otherwise noted. 

Please see the individual modules for bibliographical information
for the respective surface.

PES name        Description
==============  ==================================================
H2O_PJT1996     Water. Cartesian coordinates H--O--H.
H2O_dummy       Water. Cartesian coordinates H--O--H.
H3+_MBB1986     H3 cation. Cartesian coordinates.
CO2_ZBAR1999    CO2. Cartesian coordinates O--C--O.
Si2C            Disilicon carbide. Cartesian coordinates Si--C--Si.
H2O2_MK2012_BO  H2O2, hydrogen peroxide, internal coordinates (see notes).
H2O2_MK2012_AD  H2O2, hydrogen peroxide, internal coordinates (see notes).
==============  ==================================================


"""

