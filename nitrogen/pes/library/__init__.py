"""
nitrogen.pes.library
--------------------

A collection of potential energy surfaces. These should be accessed
by importing the `PES` object from the respective module::
    
    from nitrogen.pes.library.h2o_dummy import PES


All functions have Cartesian coordinate inputs (Angstroms) and 
return energies in cm**-1, unless otherwise noted. 

Please see the individual modules for further details, including
bibliographical information, for each surface.

================   ===============================================================
Module name        Description
================   ===============================================================
h2o_pjt1996        Water. Cartesian coordinates H--O--H.
h2o_dummy          Water. Cartesian coordinates H--O--H.
h3cation_mbb1986   H3 cation. Cartesian coordinates.
co2_zbar1999       CO2. Cartesian coordinates O--C--O.
si2c               Disilicon carbide. Cartesian coordinates Si--C--Si.
h2o2_mk2012_bo     H2O2, hydrogen peroxide, internal coordinates (see notes).
h2o2_mk2012_ad     H2O2, hydrogen peroxide, internal coordinates (see notes).
c2h_chptp2000      C2H radical, multistate diabatic surfaces for X + A states.
================   ===============================================================


"""

