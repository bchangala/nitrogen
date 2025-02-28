Geometry optimization and vibrational analysis
==============================================

In this example, we show how to optimize the
minimum energy geometry and calculate the harmonic
frequencies and normal modes of a polyatomic molecule.

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