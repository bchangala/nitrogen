..  _tut-coordsys:

Coordinate systems and the CoordSys class
=========================================

In NITROGEN, a coordinate system (CS) refers to any mapping between a set of curvilinear
coordinates and a set of Cartesian-like or rectangular coordinates (i.e., 
those with a constant, diagonal metric tensor). The most common situation for
molecular nuclear motion problems is a set of internal coordinates (like bond
lengths and angles) that defines the values of 3N atomic Cartesian coordinates,
but other CSs that do not explicitly refer to 3-D space are equally
valid. A CS is implemented with the :class:`~nitrogen.coordsys.CoordSys`
class and its sub-classes, all of which are themselves
sub-classes of :class:`~nitrogen.dfun.DFun`. 


Getting started
---------------

Let's explore :class:`~nitrogen.coordsys.CoordSys` objects with a simple
built-in CS for triatomic systems, :class:`~nitrogen.coordsys.Valence3`, 
a sub-class of :class:`~nitrogen.coordsys.CoordSys`:

..  doctest:: example-coordsys-1
    
    >>> import nitrogen as n2
    >>> cs = n2.coordsys.Valence3() 
    >>> isinstance(cs, n2.coordsys.CoordSys)
    True 

:class:`~nitrogen.coordsys.CoordSys` objects have some simple descriptor
attributes such as a name, the number of input (curvilinear) coordinates Q 
and output (rectangular) coordinates X, and coordinate labels.

..  doctest:: example-coordsys-1
    
    >>> cs.name
    'Triatomic valence'
    >>> cs.nQ, cs.nX
    (3, 9)
    >>> cs.Qstr
    ['r1', 'r2', 'theta']
    >>> cs.Xstr
    ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']

:class:`~nitrogen.coordsys.Valence3` has three input coordinates: two bond 
lengths (``r1`` and ``r2``) and a bond angle (``theta``, in radians). The 
output coordinates are the Cartesian positions of the three atoms placed at
:math:`(0,0,-r_1)`, the origin, and 
:math:`(0, r_2 \sin\theta, -r_2 \cos\theta )`, respectively. 
Their values
and derivatives are calculated with the :meth:`~nitrogen.coordsys.CoordSys.Q2X`
class method.

..  doctest:: example-coordsys-1

    >>> cs.Q2X(np.array([1.0, 1.3, 2*n2.pi/3])) # deriv = 0 by default
    array([[ 0.        ,  0.        , -1.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  1.12583302,  0.65      ]])
    >>> cs.Q2X(np.array([1.0, 1.3, 2*n2.pi/3]), deriv = 1)
    array([[ 0.        ,  0.        , -1.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  1.12583302,  0.65      ],
           [ 0.        ,  0.        , -1.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        , -0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.8660254 ,  0.5       ],
           [ 0.        ,  0.        , -0.        ,  0.        ,  0.        ,
             0.        ,  0.        , -0.65      ,  1.12583302]])



:class:`~nitrogen.coordsys.CoordSys` objects are also instances of 
:class:`~nitrogen.dfun.DFun` with the expected attributes.

..  doctest:: example-coordsys-1
    
    >>> cs.nQ == cs.nx # DFun number of inputs 
    True 
    >>> cs.nX == cs.nf # DFun number of outputs 
    True 
    
:meth:`~nitrogen.coordsys.CoordSys.Q2X` is just a wrapper for 
:meth:`DFun.f() <nitrogen.dfun.DFun.f>` and returns the same standard
derivative array. (See this section on :ref:`derivative arrays <tut-dfun-darray>`.)

Z-matrix coordinates
--------------------

The standard Z-matrix provides a useful general purpose 
internal coordinate system. It is implemented with the 
built-in CoordSys sub-class :class:`~nitrogen.coordsys.ZMAT`. Building
a Z-matrix CS requires passing a Z-matrix definition string to the 
:class:`~nitrogen.coordsys.ZMAT` initializer. This example creates a Z-matrix 
CS that is functionally equivalent to :class:`~nitrogen.coordsys.Valence3`:

..  doctest:: example-coordsys-2
    
    >>> zmat = """
    ... H 
    ... O 1 r1  
    ... H 2 r2 1 theta
    ... """
    >>> cs = n2.coordsys.ZMAT(zmat)

(Note that the default unit for :class:`~nitrogen.coordsys.ZMAT`
angles is degrees, not radians.) 
The definition string uses standard Z-matrix conventions with a few 
additional features. Any Z-matrix value with a valid variable label 
(i.e. beginning with a letter, not a number) becomes a coordinate.
The same coordinate label can be used twice, which will constrain 
multiple Z-matrix values to be equal:

    >>> zmat = """
    ... H 
    ... O 1 r   
    ... H 2 r 1 theta
    ... """
    
The order of coordinates in the CS is the order of their first appearance
in the Z-matrix definition string. A particular Z-matrix value can 
be held fixed and excluded from the CS by giving it a literal numeric value.

    >>> zmat = """
    ... H 
    ... O 1 r1   
    ... H 2 r2 1 105.0
    ... """

Coordinate labels can also be prefixed with a sign (``+`` or ``-``) or 
a constant numerical coefficient:

    >>> zmat = """
    ... H 
    ... O 1 +r   
    ... H 2 -r 1 2.3*a
    ... """
    
The atom label at the start of each line is required, but has
no meaning other than to indicate a dummy atom, for which the label is ``X``
or ``x``.

    >>> zmat = """
    ... H 
    ... O 1 r1  
    ... H 2 r2 1 a1
    ... X 3 r3 2 a2 1 tau
    ... """
    
The above Z-matrix will have six input coordinates, but generate only nine
Cartesian output coordinates for the first three atoms. The Cartesian coordinates
for dummy atoms are *not* included as output coordinates.

Coordinate transformations
--------------------------

Because :class:`~nitrogen.coordsys.CoordSys` objects are also instances of
:class:`~nitrogen.dfun.DFun`, general non-linear coordinate transformations
can be implemented using :class:`~nitrogen.dfun.DFun` composition. (See :ref:`this 
section <tut-dfun-comp>` of the DFun tutorial.) A :class:`~nitrogen.coordsys.CoordSys`
object ``XQ`` that implements :math:`X(Q)` and a :class:`~nitrogen.dfun.DFun` object 
``QQp`` that implements :math:`Q(Q')`, where :math:`Q'` are the new, transformed coordinates
can be combined to generate the transformed coordinate system :math:`X(Q(Q'))` via

    >>> XQp = QQp ** XQ # doctest: +SKIP
    >>> XQp = XQ @ QQp # functionally equivalent # doctest: +SKIP

In this case, because ``QQp`` is only an instance of :class:`~nitrogen.dfun.DFun`,
the most generic parent class, the resulting composition is also an instance of
:class:`~nitrogen.dfun.DFun`, not :class:`~nitrogen.coordsys.CoordSys`.

To formalize the special role of coordinate transformations, NITROGEN provides
yet another :class:`~nitrogen.dfun.DFun` sub-class, called
:class:`~nitrogen.coordsys.CoordTrans`, for implementing coordinate transformation 
functions. A simple and commonly used transformation is a linear one,

    :math:`Q_i = T_{ij} Q'_j`,
    
where :math:`T_{ij}` is a constant matrix. This is implemented by the built-in 
:class:`~nitrogen.coordsys.LinearTrans`, a sub-class of
:class:`~nitrogen.coordsys.CoordTrans`. As an example, let's use a linear 
transformation to construct symmetrized coordinates for water. We begin again 
with the Z-matrix CS:

..  doctest:: example-coordsys-3
    
    >>> zmat = """
    ... H 
    ... O 1 r1  
    ... H 2 r2 1 theta
    ... """
    >>> cs = n2.coordsys.ZMAT(zmat) 

We define :math:`r_s = (r_1 + r_2)/2` and :math:`r_a = (r_1 - r_2)/2`, so the 
transformation matrix is :math:`T = ((1, 1, 0), (1, -1, 0), (0, 0, 1))`

..  doctest:: example-coordsys-3 

    >>> T = np.array([[1.0, 1.0, 0], [1.0, -1.0, 0], [0, 0, 1.0]])
    >>> ct = n2.coordsys.LinearTrans(T, Qpstr = ['rs', 'ra', 'theta'], name = 'sym')
    >>> cs_sym = ct ** cs 
    >>> isinstance(cs_sym, n2.coordsys.CoordSys)
    True 

Coordinate system diagrams
--------------------------

Coordinate system and transformation objects provide a 
:meth:`~nitrogen.coordsys.CoordSys.diagram` method that generates a 
string-based pictorial representation of the CS. This can help visualize
a chain of possibly many transformations to keep track of its sequence and order.
Note that the diagram strings use unicode characters, and the appearance may
be affected by your font settings.

..  doctest:: example-coordsys-3 

    >>> print(cs_sym.diagram()) # doctest: +SKIP
         │↓              ↑│        
         │Q'[3]           │        
       ╔═╧══════╗         │        
       ║        ║         │        
       ║  sym   ║         │        
       ║        ║         │        
       ╚═╤══════╝         │        
         │Q [3]           │        
         │↓              ↑│        
         │Q [3]      [9] X│        
       ╔═╪════════════════╪═╗      
       ║ │ ┌────────────┐ │ ║      
       ║ ╰─┤  Z-matrix  ├─╯ ║      
       ║   └────────────┘   ║      
       ╚════════════════════╝  

This diagram shows that the inputs are now the new :math:`Q'` coordinates,
which are transformed to the original Z-matrix :math:`Q` coordinates, which 
are finally used to compute the output Cartesian coordinates :math:`X`.
The numbers in brackets indicate the number of coordinates in each segment.

Atomic coordinate systems
-------------------------

*Atomic* coordinate systems refer specifically to :class:`~nitrogen.coordsys.CoordSys`
objects whose output coordinates are the :math:`3N` Cartesian positions of 
:math:`N` particles in 3-D space. The coordinates must be ordered
:math:`x_0, y_0, z_0, x_1, y_1, z_1, \ldots`. The object attribute 
:attr:`~nitrogen.coordsys.isatomic` is ``True`` for atomic coordinate systems.
Certain class methods, such as :meth:`~nitrogen.coord.CoordSys.Q2g`, have 
different behavior or options for atomic vs. non-atomic coordinate systems. 
:class:`~nitrogen.coordsys.CoordSys` objects for molecular problems,
like the built-in class :class:`~nitrogen.coordsys.ZMAT`, should 
generally be atomic. The :attr:`~nitrogen.coordsys.isatomic` value of
transformed coordinate systems is inherited from that of the
untransformed coordinate sytem:

..  doctest:: example-coordsys-3 
    
    >>> cs.isatomic # instance of ZMAT
    True
    >>> cs_sym.isatomic # symmetry-transformed ZMAT coordinates
    True


