# -*- coding: utf-8 -*-
"""
Malonaldehyde potential energy surface (with automatic differentiation)

from W. Mizukami, S. Habershon, and D. P. Tew
J. Chem. Phys. 141, 144310 (2014)
doi.org/10.1063/1.4897486

The surface is partitioned into a zeroth order and correction surface

..  math ::
    V = V_0 + V_\\mathrm{corr}

The zeroth order surface is a simple sum of Morse potentials between
bonded atoms. The correction surface is a sum of distributed Gaussians
taking as arguments the distance between pairs, two pairs, or three
pairs of atoms.

This module provides two DFun objects. `PESX` is the function with 
respect to the Cartesian coordinates.
The input Cartesian coordinate (Angstrom) ordering is the same as this geometry

C            0.0035239647        0.0000000000       -1.1379095138
C            1.1859410623        0.0000000000       -0.4671627095
O            1.2951373352        0.0000000000        0.8494123252
C           -1.2326642069        0.0000000000       -0.3950488380
O           -1.2836327508        0.0000000000        0.8382056815
H           -0.0071493753        0.0000000000       -2.2163348766
H            0.3688205810        0.0000000000        1.1962401614
H           -2.1703435327        0.0000000000       -0.9688628849
H            2.1404514581        0.0000000000       -0.9796660170

`PESQ` is the function with respsect to internal coordinates, defined by 
this Z-matrix (Angstroms and degrees)

C
C 1 rCC1
O 2 rCO1 1 aO1
C 1 rCC2 2 aCCC 3 D1
O 4 rCO2 1 aO2 2 D2
H 1 rH1 2 aH1 3 D3
H 3 rOH 2 aOH 1 D4
H 4 rH2 1 aH2 6 D5
H 2 rH3 1 aH3 6 D6

"""

import nitrogen.autodiff.cyad as adc
import nitrogen as n2 
from .malpes import pes 

###################
# Create a DFun for the Cartesian PES
#
PESX = adc.ForwardDFun(pes, 1, 3*9) # The Cartesian function 
#
###################

###################
# Create a DFun for an 
# internal coordinate PES
#
zmat = """
C
C 1 rCC1
O 2 rCO1 1 aO1
C 1 rCC2 2 aCCC 3 D1
O 4 rCO2 1 aO2 2 D2
H 1 rH1 2 aH1 3 D3
H 3 rOH 2 aOH 1 D4
H 4 rH2 1 aH2 6 D5
H 2 rH3 1 aH3 6 D6
"""
cs = n2.coordsys.ZMAT(zmat)

# r0 = [1.36, 
#       1.32, 125.0, 
#       1.36, 120.0, 0.0,
#       1.32, 125.0, 0.0,
#       1.08, 120.0, 180.0,
#       0.99, 105.0, 0.,
#       1.08, 120.0, 0.,
#       1.08, 120.0, 0.,]
    
PESQ = adc.ForwardDFun(pes, 1, 3*9, input_fun = cs)
#
############################