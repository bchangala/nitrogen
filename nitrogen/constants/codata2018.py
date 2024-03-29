"""
codata2018.py

Physical constants from CODATA 2018. [1]_

References
----------
.. [1] Eite Tiesinga, Peter J. Mohr, David B. Newell, and Barry N. Taylor
   (2020), "The 2018 CODATA Recommended Values of the Fundamental Physical
   Constants" (Web Version 8.1). Database developed by J. Baker, M. Douma, 
   and S. Kotochigova. Available at http://physics.nist.gov/constants, 
   National Institute of Standards and Technology, Gaithersburg, MD 20899.
   
"""

# the constants dictionary
# key : constant string
# value: tuple with value, uncertainty, and source
#
# All values are in the [A, hc*cm^-1, u] unit system
#
# 1 A = 1 Angstrom = 1e-10 m  (exact)
# 1 hc*cm^-1 = 6.626 070 15e-34 [J s] * 299 792 458 [m s^-1] * 100 [m^-1]  (exact)
# 1 u = m(12C) / 12 = 1.660 539 066 60(50) e -27  kg  (inexact)
#
# 1 time unit = sqrt( u * A**2 / (hc*cm^-1) )
# In seconds, this is inexact because of the uncertainty in 1 u
#             
#             =  9.14294658533175854862e-13 s, inexact
# The relative uncertainty is one-half that of u/kg.
#

_kb_HzK = 2.083661912e10 # kb in Hz/K, exact
_c_SI   = 299792458.0 # c in m/s, exact
_h_SI   = 6.62607015e-34 # h in J*s, exact
_u_SI   = 1.66053906660e-27 # u in kg, inexact
_u_runc = 3.0e-10 # relative uncertainty of u in SI

_joule = 1/(_h_SI * _c_SI * 100.0)  # 1 joule in hc*cm^-1, exact
_NA = 6.02214076e23                 # Avogadro constant, exact

_t_SI   = 9.142946585331758e-13 # the Time Unit in s
_t_unc = _u_runc * 0.5 * _t_SI  # uncertainty of Time Unit in s

_h_nit = 36.483216005362721     # h in [A,hc*cm^-1, u]
_h_nit_unc = _u_runc * 0.5 * _h_nit
twopi = 6.2831853071795864769   # 2 * pi

_a0_nit = 0.529177210903
_a0_rel = 1.5e-10

_eh_nit = 2.1947463136319697e5

_c_nit = 299792458.0 * 1e10 * _t_SI
_c_nit_unc = _c_nit * _u_runc * 0.5

_Coulomb = 1/(1.602176634e-19) # 1 Coulomb in e, exact

_t_au = 2.645628850622830216331e-5
_t_runc = 1.50012e-10 # rel uncertainty in t_au, dominated by 0.5 * unc(u)

_constants = {
    # 
    # kb in Hz/K is exact
    # Conversion from Hz to hc*cm^-1 is exact
    "kb" : (_kb_HzK/(_c_SI * 100.0), 0),
    #
    # joule, exact
    # Use inverse of exact definition of hc*cm^-1 above
    "joule" : (_joule, 0),
    #
    # Use inverse of expression for time unit in s above
    "second" : ( 1/_t_SI,  _u_runc * 0.5 / _t_SI),
    #
    # Planck constant
    "h" : (_h_nit, _h_nit_unc),
    "hbar" : (_h_nit / twopi, _h_nit_unc / twopi),
    
    # Bohr radius
    "a0" : (_a0_nit, _a0_rel * _a0_nit),
    
    # electron-volt, exact
    "eV" : (8.065543937e3, 0),
    
    # hartree, inexact
    "Eh" : (_eh_nit, _eh_nit * 1.9e-12),
    
    # speed of light, inexact
    "c" : (_c_nit, _c_nit_unc),
    
    # m_e, electron mass (in u)
    "me" : (5.48579909065e-4, 0.16e-13),
    
    # Debye, exact (in e * Angstrom)
    "debye" : (1e-11 /(299792458.0 * 1.602176634e-19), 0),
    
    # Avogadro constant, exact
    "NA" : (6.02214076e23, 0),
    
    # kilojoule / mol, exact
    "kJ" : (1000.0 * _joule/_NA, 0),
    
    # kilocalorie / mol, exact
    "kcal": (4184.0 * _joule/_NA, 0),
    
    # atomic unit of time, inexact
    "t_au" : (_t_au, _t_au * _t_runc),
    
    # Frequency-energy equivalent (exact)
    "hHz" : (1e-6/29979.2458, 0),
    
    # electron-proton mass ratio, dimensionless
    "memp" : (5.44617021487e-4, 3.3e-14),
    
    # proton mass, in u 
    "mp"   : (1.007276466621, 5.3e-11),
    
    }