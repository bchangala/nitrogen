Geometry optimization and vibrational analysis
==============================================

In this example, we show how to optimize the
minimum energy geometry and calculate the harmonic
frequencies and normal modes of a polyatomic molecule.

Geometry optimization
---------------------

We first need a potential energy surface (PES) to work with. 
We will use one for H\ :sub:`2`\ O published by 
Polyansky et al (J. Chem. Phys. 105, 6490, 1996), which is 
available as a built-in :class:`~nitrogen.dfun.DFun` object.

..  doctest:: example-geoopt

    >>> import nitrogen as n2 
    >>> from nitrogen.pes.library.h2o_pjt1996 import PES 
    >>> PES.nx
    9 
    
The ``PES`` object returns the potential energy surface 
(in cm\ :sup:`-1`) as a 
function of the 9 Cartesian coordinates (in Angstroms)
in H, O, H order.

We wish to perform a geometry optimization with respect 
to curvilinear internal coordinates. To do so, we must first 
define a coordinate system transforming between internal and 
Cartesian coordinates, and then construct a PES with internal coordinate 
arguments.

..  doctest:: example-geoopt

    >>> zmat = """
    ... H 
    ... O 1 r1  
    ... H 2 r2 1 theta
    ... """
    >>> cs = n2.coordsys.ZMAT(zmat)
    >>> V = cs ** PES 
    

This code snippet defines the internal Z-matrix coordinate system
in terms of the two OH bond lengths and HOH bond angle and then
defines ``V`` as a new :class:`~nitrogen.dfun.DFun` that accepts
internal coordinate arguments, transforms these to Cartesians, and then 
evaluates the original ``PES`` function.  

NITROGEN has various optimizer routines. A good general purpose 
optimizer is :py:func:`nitrogen.pes.opt_bfgs`, which is a standard 
Broyden--Fltetcher--Goldfarb--Shanno (BFGS) algorithm. Let's call it
with an initial geometry of r(OH) = 1.0 Å and :math:`\theta`\ (HOH)
= 100\ :math:`^\circ` and extra information printed. 

..  doctest:: example-geoopt

    >>> qmin,Vmin = n2.pes.opt_bfgs(V, [1.0, 1.0, 100.0], disp = True)
    Step   Value       |grad|         
    -----------------------------
      1    7.0178e+02  2.0302e+04  ...  [  1.   1. 100.]
      2    1.9115e+01  4.0473e+03  ...  [  0.95115476   0.95115476 104.70196102]
      3    4.2499e-01  5.9103e+02  ...  [  0.95893928   0.95893928 104.46238834]
      4    2.2943e-04  1.3751e+01  ...  [  0.95794415   0.95794415 104.49873555]
      5    2.8546e-09  4.8817e-02  ...  [  0.95792042   0.95792042 104.49964887]
      6    3.6116e-15  3.6321e-06  ...  [  0.9579205    0.9579205  104.49964697]
      7    1.3489e-18  8.6905e-08  ...  [  0.9579205   0.9579205 104.499647 ]
    <BLANKLINE>
    Convergence reached, |g| = 8.690e-08
    7 gradient(s) and 1 Hessian(s) were calculated.
    >>> qmin # Minimum energy geometry (rOH, rOH, aHOH)
    array([  0.9579205,   0.9579205, 104.499647 ])
    >>> Vmin # Minimum energy value (cm^-1)
    1.3489154554107497e-18

:py:func:`nitrogen.pes.opt_bfgs` has various options for changing the convergence
tolerance, constraining one or more arguments to fixed values, supplying a pre-computed
Hessian, etc. 

Curvilinear vibrational analysis
--------------------------------

Now that we have optimized the equilibrium geometry, we can perform a 
harmonic vibrational normal-mode analysis. We will first do this explicity using 
our Z-matrix coordinate system using the standard curvilinear GF approach (see 
Wilson, Decius, and Cross 1955). Before doing so, we need to define the atomic 
masses.

..  doctest:: example-geoopt

    >>> masses = n2.constants.mass(['H','O','H'])
    >>> masses 
    [1.00782503224, 15.9949146196, 1.00782503224]
    >>> omega, nctrans = n2.pes.curvVib(qmin, V, cs, masses)
    
:py:func:`nitrogen.pes.curvVib` returns harmonic frequencies and a 
linear transformation object containing the normal-mode displacement vectors.

..  doctest:: example-geoopt

    >>> omega # harmonic frequencies (* hc, in cm-1)
    array([1649.58906249, 3830.38088976, 3940.96386738])
    >>> nctrans.T  # columns are the normal-coordinate displacement vectors 
    array([[ 7.11357961e-03,  6.74635431e-02,  6.76662154e-02],
           [ 7.11357961e-03,  6.74635431e-02, -6.76662154e-02],
           [-1.25106206e+01,  9.57536309e-02, -8.98152710e-16]])
           
The displacement vectors are scaled to equal reduced dimensionless normal 
coordinates, :math:`q`, i.e., the coordinates in which the harmonic potential 
is :math:`V = \frac{1}{2} \omega q^2`, where :math:`\omega` is the harmonic 
frequency (in energy units). 

A new coordinate system can be constructed using these curvilinear normal coordinates. 
Let's build this and verifying that the equilibrium geometry is :math:`q = 0` and 
that the Hessian in this coordinate system is diagonal with elements equal to 
the harmonic frequencies. 

..  doctest:: example-geoopt

    >>> cs2 = nctrans ** cs # The normal-mode coordinate system 
    >>> V2 = nctrans ** V  # The PES w.r.t normal-mode coordinates (q)
    >>> qmin,Vmin = n2.pes.opt_bfgs(V2, [0.1, 0.2, 0.3], disp = True)
    Step   Value       |grad|         
    -----------------------------
      1    2.4367e+02  1.2706e+03  ...  [0.1 0.2 0.3]
      2    4.3457e+00  1.8472e+02  ...  [-0.00926409 -0.0339949  -0.03199741]
      3    6.3725e-02  2.2157e+01  ...  [0.00097352 0.00393639 0.00411375]
      4    1.5009e-05  3.4152e-01  ...  [9.69510200e-06 5.54551059e-05 6.77409462e-05]
      5    2.1711e-10  1.2347e-03  ...  [-1.98193817e-07 -3.00724345e-07  7.64542669e-08]
      6    2.4936e-13  4.2101e-05  ...  [-6.66707056e-09 -7.97745234e-09  6.80267296e-09]
      7    2.0227e-18  1.2603e-07  ...  [-4.31359815e-11  1.06617424e-11  3.09024816e-11]
    <BLANKLINE>
    Convergence reached, |g| = 1.260e-07
    7 gradient(s) and 1 Hessian(s) were calculated.
    >>> hes = V2.hes(qmin)[0] 
    >>> hes 
    array([[ 1.64958906e+03,  3.39518280e-09, -7.26838984e-09],
           [ 3.39518280e-09,  3.83038089e+03, -5.39473322e-08],
           [-7.26838984e-09, -5.39473322e-08,  3.94096387e+03]])
    >>> np.allclose(np.diag(hes), omega) # diagonal elements equal omega? 
    True
            
Everything checks out.

Rectilinear vibrational analysis
--------------------------------

The standard Watson Hamiltonian is based on rectilinear normal
coordinates. These can be calculated using 
:py:func:`nitrogen.vpt.calc_rectilinear_modes`, which first requires
calculating the Hessian with respect to Cartesian displacements. To do that,
we first evaluate the equilibrium Cartesian position using our curvilinear
equilibrium geometry from above and rotate it to the principal axis system.


..  doctest:: example-geoopt

    >>> Xe = cs2.Q2X(qmin)[0] # Cartesian equilibrium geometry 
    >>> Xe,_,_ = n2.angmom.X2PAS(Xe, masses) # Rotate to PAS
    >>> hes = PES.hes(Xe)[0] # The Cartesian Hessian at Xe 
    >>> omega_rect, T = n2.vpt.calc_rectilinear_modes(hes, masses)
    >>> omega_rect
    array([5.00232981e-03, 1.36059436e-05, 7.38904032e-06, 2.88647058e-05,
           2.93979778e-03, 6.45802187e-03, 1.64958906e+03, 3.83038089e+03,
           3.94096387e+03])
           
The harmonic frequencies in ``omega_rect`` include the 3 translational and 
3 rotational modes, which equal zero. The vibrational frequencies equal
the those calculated with the curvilinear GF method above.

The normal-mode Cartesian displacement vectors are returned as the columns of 
``T``.  By default, :py:func:`~nitrogen.vpt.calc_rectilinear_modes` normalizes
the vibrational vectors to the same reduced dimensionless normal coordinates, 
:math:`q`, as above. 
To request the displacement vectors with respect to mass-weighted Cartesians,
use the `norm` keyword. These are normalized to unity modulus.

..  doctest:: example-geoopt

    >>> omega_rect, L = n2.vpt.calc_rectilinear_modes(hes, masses, norm = 'mass-weighted')
    >>> np.allclose(L.T @ L, np.eye(len(L))) # L is orthonormal
    True
    
The mass-weighted displacement vectors can be used to calculate Coriolis
coupling constants

..  doctest:: example-geoopt

    >>> Lvib = L[:,6:] # The vibrational vectors 
    >>> zeta = n2.vpt.calc_coriolis_zetas(Lvib) 
    >>> zeta # (mode i, mode j, axis k) 
    array([[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 7.21324074e-17, -2.51543557e-17, -2.54098520e-10],
            [-7.78383835e-17,  1.35562504e-16,  9.99931994e-01]],
    <BLANKLINE>
           [[-7.21324074e-17,  2.51543557e-17,  2.54098520e-10],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-9.08401587e-17, -1.21849559e-16,  1.16621858e-02]],
    <BLANKLINE>
           [[ 7.78383835e-17, -1.35562504e-16, -9.99931994e-01],
            [ 9.08401587e-17,  1.21849559e-16, -1.16621858e-02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])
            
Harmonic centrifugal distortion constants can also be calculated

..  doctest:: example-geoopt 
    
    >>> B0,CD = n2.vpt.analyzeCD(Xe, omega_rect[6:], Lvib, masses, printing = True)
    ==========================================
     Harmonic centrifugal distortion analysis 
    ==========================================
    <BLANKLINE>
    <BLANKLINE>
                   cm^-1             MHz      
              --------------    --------------
       Ae       2.73812E+01      820869.10098   
       Be       1.45785E+01      437052.06382   
       Ce       9.51334E+00      285202.71759   
    <BLANKLINE>
       A'       2.73832E+01      820927.74239   
       B'       1.45804E+01      437110.70523   
       C'       9.51040E+00      285114.75547   
    <BLANKLINE>
     A' - Ae    1.95607E-03          58.64141   
     B' - Be    1.95607E-03          58.64141   
     C' - Ce   -2.93410E-03         -87.96212   
    <BLANKLINE>
              ------------- (Ir) -------------
       A(A)     2.73816E+01      820878.74053   
       B(A)     1.45862E+01      437283.44557   
       C(A)     9.50628E+00      284991.01699   
    <BLANKLINE>
     A(A)-Ae    3.21541E-04           9.63955   
     B(A)-Be    7.71806E-03         231.38175   
     C(A)-Ce   -7.06157E-03        -211.70060   
    <BLANKLINE>
       A(S)     2.73817E+01      820883.30410   
       B(S)     1.45854E+01      437259.53118   
       C(S)     9.50695E+00      285011.28053   
    <BLANKLINE>
     A(S)-Ae    4.73765E-04          14.20312   
     B(S)-Be    6.92037E-03         207.46736   
     C(S)-Ce   -6.38565E-03        -191.43706   
    <BLANKLINE>
              ------------ (IIIr) ------------
       A(A)     2.73717E+01      820582.11772   
       B(A)     1.46211E+01      438329.09419   
       C(A)     9.48129E+00      284241.99118   
    <BLANKLINE>
     A(A)-Ae   -9.57273E-03        -286.98326   
     B(A)-Be    4.25971E-02        1277.03036   
     C(A)-Ce   -3.20464E-02        -960.72640   
    <BLANKLINE>
       A(S)     2.73974E+01      821351.94126   
       B(S)     1.45755E+01      436963.44257   
       C(S)     9.50614E+00      284986.77628   
    <BLANKLINE>
     A(S)-Ae    1.61058E-02         482.84029   
     B(S)-Be   -2.95609E-03         -88.62125   
     C(S)-Ce   -7.20303E-03        -215.94131   
    <BLANKLINE>
       sigma ............ 6.050359         
    <BLANKLINE>
      ---------------------------------------  
             Kivelson-Wilson parameters        
      ---------------------------------------  
          DJ    9.59431E-04          28.76303   
          DK    2.47000E-02         740.48861   
         DJK   -3.75105E-03        -112.45370   
          R5    1.05198E-03          31.53761   
          R6   -1.02158E-04          -3.06262   
    <BLANKLINE>
      ---------------------------------------  
              A-reduced (Ir) parameters        
      ---------------------------------------  
      DeltaJ    1.16375E-03          34.88826   
      DeltaK    2.57216E-02         771.11477   
     DeltaJK   -4.97695E-03        -149.20510   
      deltaJ    4.64437E-04          13.92346   
      deltaK    3.68404E-04          11.04448   
    <BLANKLINE>
      ---------------------------------------  
              S-reduced (Ir) parameters        
      ---------------------------------------  
          DJ    1.13330E-03          33.97555   
          DK    2.55694E-02         766.55120   
         DJK   -4.79428E-03        -143.72882   
          d1   -4.64437E-04         -13.92346   
          d2   -1.52224E-05          -0.45636   
    <BLANKLINE>
      ---------------------------------------  
              A-reduced (IIIr) parameters      
      ---------------------------------------  
      DeltaJ    1.20005E-02         359.76656   
      DeltaK    2.57216E-02         771.11477   
     DeltaJK   -3.74873E-02       -1123.84000   
      deltaJ    4.95395E-03         148.51569   
      deltaK   -1.78079E-02        -533.86879   
    <BLANKLINE>
      ---------------------------------------  
              S-reduced (IIIr) parameters      
      ---------------------------------------  
          DJ    7.03185E-03         210.80954   
          DK    8.78263E-04          26.32968   
         DJK   -7.67524E-03        -230.09788   
          d1   -4.95395E-03        -148.51569   
          d2   -2.48434E-03         -74.47851   
    <BLANKLINE>
    ==========================================

    

