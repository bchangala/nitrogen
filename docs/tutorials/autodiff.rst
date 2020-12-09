..  _tut-autodiff:

Automatic differentiation with the ``autodiff`` sub-package
===========================================================

`Automatic differentiaton <https://en.wikipedia.org/wiki/Automatic_differentiation>`_
(AD) refers to a diverse set of techniques used to numerically evaluate the 
derivatives of arbitrary (analytical) functions. NITROGEN requires access 
to such derivatives for a variety of tasks. Although the :class:`~nitrogen.dfun.DFun`
class defines an interface to differentiable functions and some high-level
operations such as composition, it does not prescribe an implementation.
In many built-in NITROGEN objects, derivatives are computed via "forward
accumulation" AD module (:mod:`~nitrogen.autodiff.forward`) of the 
:mod:`nitrogen.autodiff` sub-package. This chapter provides a guide to 
the design and usage of this module.

A brief introduction to AD
--------------------------

Forward AD relies on the accumulative use of the chain rule, product rule, and
Taylor series to evaluate the derivatives of analytical expressions in terms of the
derivatives of their arguments. Given an expression :math:`f` with 
derivatives :math:`\partial^\alpha f` (using :ref:`multi-index notation <tut-dfun-darray>`),
we begin with some simple rules from the linearity of differentiation:

..  math::

    w &= af &\rightarrow \partial^\alpha w &= a \left(\partial^\alpha f\right)
    
    w &= f + g &\rightarrow \partial^\alpha w &= \partial^\alpha f + \partial^\alpha g,

where :math:`a` is a scalar constant. It is convenient to introduce a scaled
derivative,

..  math::

    f^{(\alpha)} = \frac{1}{\alpha !} \partial^\alpha f,
    
which renders the above rules unchanged,

..  math::

    (af)^{(\alpha)}&= a  f^{(\alpha)}
    
    (f + g)^{(\alpha)} &= f^{(\alpha)} + g^{(\alpha)}.
    
Another fundamental result is the generalized product rule, or Leibniz
formula, for multi-variable functions,

..  math::

    (fg)^{(\alpha)} = \sum_{\beta \leq \alpha} f^{(\beta)} g^{(\alpha-\beta)},
    
where :math:`\beta \leq \alpha` if and only if :math:`\beta_i \leq \alpha_i`
for all :math:`i`. (The use of scaled derivatives removes multi-index
binomial coefficients otherwise present in this formula.) The Leibniz formula 
implies that product derivatives up to a given order :math:`k = \vert \alpha \vert`
only require the derivatives of the factors up to the same order.

Finally, we also need to consider chain rule or function composition expressions,
:math:`h(x) = f(g(x))`. Explicit formulae, such as that of
`Faa di Bruno <https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula>`_,
provide formal expressions relating the derivatives of :math:`h` to those of
:math:`f` and :math:`g` with their respective arguments. These
expressions, however, are somewhat cumbersome to implement directly. Instead,
we rely on a Taylor series approach. The multi-variate function :math:`f(y)`
can be expanded about the value :math:`y_0` as 

..  math::

    f(y) = \sum_\beta f^{(\beta)} \vert_{y=y_0} (y - y_0)^\beta 

If we define :math:`h(x) = f(g(x))`, then we can expand :math:`h` about
:math:`x_0` as 

..  math::

    h(x) = \sum_\beta f^{(\beta)} \vert_{y=g(x_0)} (g - g_0)^\beta 
    
Assuming all functions are sufficiently well behaved, it can be shown that
the truncated series

..  math::

    h_k(x) \equiv \sum_{\vert \beta \vert \leq k} f^{(\beta)} \vert_{y=g(x_0)} (g - g_0)^\beta 
    
has the same derivatives as :math:`h(x)` at :math:`x = x_0` up to the expansion order, i.e.

..  math::
    h_k^{(\alpha)} \vert_{x_0} = h^{(\alpha)} \vert_{x_0} \text{ for } \vert \alpha \vert \leq k.
    
The derivatives of :math:`f` and :math:`g` are assumed to be known, so repeated application
of the Leibniz formula allows one to straightforwardly compute the truncated Taylor series.
This procedure is the basis by which :mod:`~nitrogen.autodiff` implements most 
mathematical functions and including trigonometric, exponential, and logarithmic functions.

Working with ``adarray`` objects
--------------------------------

NITROGEN implements the concepts introduced above with the :class:`~nitrogen.autodiff.forward.adarray`
class. An instance of this class stores the derivatives for an expression with 
respect to a given number of variables up to a fixed order. To illustrate how to 
use this class, let's construct the expression :math:`f = 3 + 2 x^2 - 4(x-y)`,
where :math:`x` and :math:`y` are the two independent variables. 

First, we create the :class:`~nitrogen.autodiff.forward.adarray` objects for the 
two independent variables ("symbols") with the :func:`~nitrogen.autodiff.forward.sym` 
function:

..  doctest:: example-adf-1

    >>> import nitrogen.autodiff.forward as adf
    >>> x = adf.sym(1.0, 0, 3, 2) # x <-- 1.0
    >>> y = adf.sym(2.5, 1, 3, 2) # y <-- 2.5

Note that the value of the symbol, the symbol index, the maximum derivative 
order, and the total number of symbols must all be specified upon construction and 
cannot be changed. The basic arithmetic operators ``+``, ``-``, ``*``, and ``/``
are overloaded for :class:`~nitrogen.autodiff.forward.adarray` operands, so 
we can evaluate :math:`f` simply as

..  doctest:: example-adf-1

    >>> f = 3 + 2 * x * x - 4 * (x - y)
   
The numerical values of the derivatives are stored in an 
:class:`ndarray` referred to by the 
:attr:`~nitrogen.autodiff.forward.adarray.d` attribute of the
:class:`~nitrogen.autodiff.forward.adarray`. The derivatives are stored with 
their scaled values using the same lexical ordering as :class:`~nitrogen.dfun.DFun`
derivative arrays. Let's inspect the contents of the derivative arrays we created:

..  doctest:: example-adf-1

    >>> x.d
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> y.d 
    array([2.5, 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])
    >>> f.d 
    array([11.,  0.,  4.,  2.,  0.,  0.,  0.,  0.,  0.,  0.])
    
The derivative arrays for symbols are very simple. The first element is the 
value of the symbol. The only other non-zero element is the first derivative 
with respect to itself, which is always 1. By inspection of ``f.d`` we can see
that it contains the correct (scaled) derivatives for :math:`f`.

As noted above, the symbol values must be specified when their
:class:`~nitrogen.autodiff.forward.adarray` objects are created. Instead of
a single value, an arbitrarily shaped :class:`~nitrogen.autodiff.forward.ndarray`
(or array-like object) can be passed for vectorized processing. This
"base-shape" must be the same for each :class:`~nitrogen.autodiff.forward.adarray`
used together.

..  doctest:: example-adf-1

    >>> x = adf.sym(np.linspace(0,1,5), 0, 3, 2) # x <-- [0, 0.25, 0.50, 0.75, 1.0]
    >>> y = adf.sym(np.linspace(0,2,5), 1, 3, 2) # y <-- [0, 0.5,  1.0,  1.5,  2.0]
    >>> x.d # shape = (nd,) + base_shape
    array([[0.  , 0.25, 0.5 , 0.75, 1.  ],
           [1.  , 1.  , 1.  , 1.  , 1.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  ]])
    >>> g = x*x*y*y
    >>> g.d
    array([[0.      , 0.015625, 0.25    , 1.265625, 4.      ],
           [0.      , 0.125   , 1.      , 3.375   , 8.      ],
           [0.      , 0.0625  , 0.5     , 1.6875  , 4.      ],
           [0.      , 0.25    , 1.      , 2.25    , 4.      ],
           [0.      , 0.5     , 2.      , 4.5     , 8.      ],
           [0.      , 0.0625  , 0.25    , 0.5625  , 1.      ],
           [0.      , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 1.      , 2.      , 3.      , 4.      ],
           [0.      , 0.5     , 1.      , 1.5     , 2.      ],
           [0.      , 0.      , 0.      , 0.      , 0.      ]])

