Differentiable functions and the DFun class
===========================================

Potential energy surfaces, coordinate systems, and coordinate
transformations are the building blocks of nuclear motion calculations.
For many of the tools available in NITROGEN, e.g., geometry optimization,
force field and reaction path analysis, and kinetic energy operator construction,
it is often necessary to evaluate these objects and their (potentially
high-order) derivatives as functions of general nuclear coordinates.
NITROGEN implements the :class:`nitrogen.dfun.DFun` class as a general interface
for such differentiable functions. Instances of this class provide a method
(:meth:`DFun.f <nitrogen.dfun.DFun.f>`) that returns the value and derivatives
of a function up to a given order in a standard array format. Multiple
:class:`~nitrogen.dfun.DFun` instances can be linked via functional composition
using the ``**`` and ``@`` operators. Many built-in NITROGEN objects are 
instances of the :class:`~nitrogen.dfun.DFun` class or its sub-classes, and 
constructing these objects is the first step for a variety of tasks.

.. _tut-dfun-darray:

The derivative array and lexical ordering
-----------------------------------------

Let :math:`f(x)` be a multi-variable differentiable function with 
:math:`n` input values :math:`x = \{x_1, x_2, \ldots, x_n \}`. 
The partial derivative with respect to :math:`x_i` is abbreviated
:math:`\partial_i = \frac{\partial}{\partial x_i}`. Given that
the order of differentiation does not matter, it is convenient to use 
`multi-index notation 
<https://en.wikipedia.org/wiki/Multi-index_notation>`_ to specify uniquely general high-order 
derivatives. A multi-index :math:`\alpha = (\alpha_1, \alpha_2, \ldots, \alpha_n)` 
contains :math:`n` non-negative integer elements. The expression :math:`x^\alpha` is equal to 
:math:`x_1^{\alpha_1} x_2^{\alpha_2} \cdots x_n^{\alpha_n}`. Thus, an arbitrary
derivative can be expressed as 

.. math::

   \partial^\alpha f = \partial_1^{\alpha_1} \partial_2^{\alpha_2} \cdots \partial_n^{\alpha_n} f. 

These derivatives could be stored numerically in 
an :math:`n`-dimensional array, but this scheme ignores the fact that one
generally needs derivatives up to a certain total order :math:`\vert \alpha \vert 
= \alpha_1 + \alpha_2 + \cdots + \alpha_n`. Instead, let us list the derivatives 
in a one-dimensional array with a particular lexical ordering scheme. Derivatives 
are first sorted by their total order :math:`\vert \alpha \vert` beginning with
the zeroth derivative (the function value), first derivatives, second derivatives, 
and so on. Within a given order, derivatives are sorted by the degree of 
:math:`\partial_1` (i.e. :math:`\alpha_1`), then by the degree of :math:`\partial_2`,
and so on. For example, the derivative array for :math:`n = 3` would be sorted 

.. math::
   \{ f, \partial_1 f, \partial_2 f, \partial_3 f, \partial_1^2 f,\partial_1 \partial_2 f, \partial_1 \partial_3f , \partial_2^2 f, \partial_2 \partial_3 f, \partial_3^2 f\}

up to second order. A combinatoric
analysis shows that there are

    :math:`\binom{k + n -1}{k}` unique derivatives of a given order :math:`\vert \alpha \vert = k`,
    
    :math:`\binom{k + n -1}{k-1}` derivatives with :math:`\vert \alpha \vert < k`, and therefore
    
    :math:`\binom{k + n}{k}` derivatives in total with :math:`\vert \alpha \vert \leq k`.

The function :func:`nitrogen.dfun.nderiv` computes this final binomial coefficient. 
It turns out to be more efficient to store derivatives including a permutational
factor

.. math::
   f^{(\alpha)} \equiv \frac{1}{\alpha !} \partial^\alpha f,
   
where :math:`\alpha ! = \alpha_1 ! \alpha_2 ! \cdots \alpha_n !`.
:class:`~nitrogen.dfun.DFun` derivative arrays contain these scaled derivatives 
and *not* the raw derivatives :math:`\partial^\alpha f`.

Creating :class:`~nitrogen.dfun.DFun` objects
---------------------------------------------

A custom :class:`~nitrogen.dfun.DFun` object requires that we first define a function 
that returns the derivative array up to a given order for a given set of 
input variables. This function should have the signature ``fx(X, deriv = 0, 
out = None, var = None)``. `X` is a (:math:`n`,...) :class:`numpy.ndarray` 
containing the value of each of the :math:`n` input variables. The remaining 
dimensions can have arbitrary shape for efficient vectorization. The optional
`deriv` parameter specifies the maximum derivative order requested, while the
`var` parameter is an ordered list of the input variables with respect to which the 
derivatives are to be calculated. ``None`` is equivalent to `var` = [0, 1, 2, ...].

Although the previous section considered only scalar-valued differentiable 
functions, :class:`~nitrogen.dfun.DFun` objects support vector-valued functions 
with :math:`n_f` output elements. ``fx`` supplies the derivatives for each of 
these :math:`n_f` outputs in a (:math:`n_d`, :math:`n_f`, ...) :class:`numpy.ndarray`, where
the trailing dimensions must match those of `X`.
:math:`n_d` is the number of derivatives, equal to ``nderiv(deriv,len(var))``.
This first index is sorted via the lexical ordering introduced above. The priority
of each variable, however, is determined by its position in the `var` list, *not*
the ordering of the input array `X`.

By default, a new output :class:`numpy.ndarray` is allocated and returned. If the 
optional `out` argument is not ``None``, however, than this should be a properly
shaped :class:`numpy.ndarray` where the output will be stored.  

Let's take a look at an example of defining ``fx`` manually:

..  testcode:: example-dfun-1

    import nitrogen as n2 
    import numpy as np 
    
    def fx(X, deriv = 0, out = None, var = None):
        """ An example DFun evaluation function implementing
            f = 3 + 5*x0 + x0*x0 + 7*x1*x1
        """
        
        # Process var parameter
        if var is None:
            var = [0, 1]
        # Calculate the number of derivatives 
        nd = n2.dfun.nderiv(deriv, len(var))
        
        # Allocate output
        if out is None: 
            out = np.ndarray( (nd, 1) + X.shape[1:], dtype = X.dtype)
        out.fill(0.0) # Initialize to zero
        
        x0 = X[0]
        x1 = X[1]
        
        one = np.ones(X.shape[1:], dtype = X.dtype)
        zero = np.zeros(X.shape[1:], dtype = X.dtype)
        
        # Calculate the function value
        f = 3.0 + 5.0*x0 + x0*x0 + 7.0*x1*x1
        
        # Calculate derivatives
        f0 = 5.0 + 2.0 * x0 # (1,0)
        f1 = 14.0 * x1      # (0,1)
        f00 = one           # (2,0) = 2 * 1 / 2!
        f01 = zero          # (1,1)
        f11 = 7.0 * one     # (0,2) = 2 * 7 / 2! 
                            # (note permutational factors!)
        # all higher-order derivatives are zero 
        
        # Copy derivatives to the 
        # properly ordered output array 
        #
        np.copyto(out[0,0:1], f) # function value 
        if var == []:
            # No variables requested
            # Only the function value is required
            pass
        elif var == [0]:
            # x0 only
            if deriv >= 1:
                np.copyto(out[1,0:1], f0)
            if deriv >= 2:
                np.copyto(out[2,0:1], f00)
        elif var == [1]:
            # x1 only
            if deriv >= 1:
                np.copyto(out[1,0:1], f1)
            if deriv >= 2:    
                np.copyto(out[2,0:1], f11)
        elif var == [0,1]:
            # Both variables in order x0, x1
            if deriv >= 1:
                np.copyto(out[1,0:1], f0)
                np.copyto(out[2,0:1], f1)
            if deriv >= 2:
                np.copyto(out[3,0:1], f00)
                np.copyto(out[4,0:1], f01)
                np.copyto(out[5,0:1], f11)
        elif var == [1,0]:
            # Both variables in order x1, x0
            if deriv >= 1:
                np.copyto(out[1,0:1], f1)
                np.copyto(out[2,0:1], f0)
            if deriv >= 2:
                np.copyto(out[3,0:1], f11)
                np.copyto(out[4,0:1], f01)
                np.copyto(out[5,0:1], f00)
        
        return out

We can now initialize a :class:`~nitrogen.dfun.DFun` object and call its
:meth:`~nitrogen.dfun.DFun.f` method:

..  doctest:: example-dfun-1

    >>> df = n2.dfun.DFun(fx, nf=1, nx=2, maxderiv=None, zlevel=2)
    >>> df.f(np.array([1.,2.]), deriv = 3)
    array([[37.],
           [ 7.],
           [28.],
           [ 1.],
           [ 0.],
           [ 7.],
           [ 0.],
           [ 0.],
           [ 0.],
           [ 0.]])
    >>> df.f(np.array([1.,2.]), deriv = 3, var = [1,0])
    array([[37.],
           [28.],
           [ 7.],
           [ 7.],
           [ 0.],
           [ 1.],
           [ 0.],
           [ 0.],
           [ 0.],
           [ 0.]])

The ``nf`` and ``nx`` options are the number of output functions and input
variables, respectively. If the ``fx`` function can only provide valid
derivatives up to some maximum order, then this limit can be specified 
with the ``maxderiv`` option. In this case, our ``fx`` implementation is 
valid for all derivative orders, so we let `maxderiv` equal ``None`` (the 
default value). Similarly, if the function is guaranteed to have no non-zero 
derivatives above a certain order, that can also be specified with the `zlevel`
option. Our simple polynomial example is quadratic, which we indicate with 
``zlevel = 2``. The default behavior of ``zlevel = None`` indicates that no 
derivatives are guaranteed to be zero. The :meth:`DFun.f <nitrogen.dfun.DFun.f>`
method is a wrapper for the supplied function ``fx``. It performs argument checks
before calling ``fx``, which is stored as a private attribute.

Manually implementing the ``fx`` function even for simple 
functions can be cumbersome. More complicated functions quickly become intractable.
NITROGEN provides a few tools for implementing :class:`~nitrogen.dfun.DFun` objects,
including numerical (finite difference) differentiation and automatic differentiation.
Of course, the user is free to use whatever backend implementation they wish as long
as it is wrapped by an approriate Python ``fx`` function.

Finite differences
~~~~~~~~~~~~~~~~~~

If the user has a function that only provides an output value, and not derivatives, 
then a :class:`~nitrogen.dfun.FiniteDFun` object can be created that approximates
its derivatives via finite differences up to order ``maxderiv = 2``. 
To implement the above example, we use:

..  testcode:: example-dfun-fd

    def fx_fd(X):
        """ An example FiniteDFun evaluation function implementing
            f = 3 + 5*x0 + x0*x0 + 7*x1*x1
        """
        
        x0 = X[0]
        x1 = X[1]
        
        # Calculate and return the function value
        return 3.0 + 5.0*x0 + x0*x0 + 7.0*x1*x1

..  doctest:: example-dfun-fd 

    >>> df = n2.dfun.FiniteDFun(fx_fd, 2)
    >>> df.f(np.array([1.,2.]), deriv = 2)
    array([[37.],
           [ 7.],
           [28.],
           [ 1.],
           [ 0.],
           [ 7.]])
       
Automatic differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~
 
The sub-package :mod:`nitrogen.autodiff.forward` implements forward-type
automatic differentiation. A detailed guide to this sub-package is deferred 
until :ref:`this chapter <tut-autodiff>`. To demonstrate its usefulness here, however, 
the following code snippet implements the same differentiable function as above using
the ``autodiff`` API. The functions :func:`nitrogen.dfun.X2adf`
and :func:`nitrogen.dfun.adf2array` are also used to convert :class:`~numpy.ndarray`
objects to :class:`~nitrogen.autodiff.forward.adarray` objects and *vice versa*.

..  testcode:: example-dfun-adf

    import nitrogen as n2 
    import nitrogen.autodiff.forward as adf
    import numpy as np 
    
    def fx_adf(X, deriv = 0, out = None, var = None):
        """ An example DFun evaluation function implementing
            f = 3 + 5*x0 + 7*x1*x1
            using the nitrogen.autodiff.forward module
        """
        
        # Create a list of adarray objects
        x = n2.dfun.X2adf(X, deriv, var)
        
        # Compute the function
        f = 3.0 + 5.0*x[0] + x[0]*x[0] + 7.0*x[1]*x[1]
        
        # Convert the adf result to a raw derivative array
        return n2.dfun.adf2array([f], out)

..  doctest:: example-dfun-adf

    >>> df = n2.dfun.DFun(fx_adf, nf=1, nx=2, maxderiv=None, zlevel=2)
    >>> df.f(np.array([1.,2.]), deriv = 3)
    array([[37.],
           [ 7.],
           [28.],
           [ 1.],
           [ 0.],
           [ 7.],
           [ 0.],
           [ 0.],
           [ 0.],
           [ 0.]])
                   
Composition of differentiable functions
---------------------------------------

Function composition is a common procedure when dealing with multiple 
coordinate systems and transformations between them. Two functions, :math:`f(y)` 
and :math:`g(x)`, can be composed to generate a new function :math:`h(x) = 
(f \circ g)(x) = f(y=g(x))`. The derivatives of :math:`h(x)` up to a given order 
are completely determined by those of :math:`f(y)` and :math:`g(x)` up to the 
same order. Given :class:`~nitrogen.dfun.DFun` objects for :math:`f(y)` and 
:math:`g(x)`, we can construct a :class:`~nitrogen.dfun.DFun` object for
:math:`h(x)` using either of the composition operators ``**`` or ``@``::
    
    h = f @ g  # h(x) = f(g(x))
    h = g ** f # equivalent

Each of these statements performs the same function composition. The ``@``
and ``**`` operators act in an "outside in" and "inside out" direction,
respectively. Both return an instance of :class:`~nitrogen.dfun.CompositeDFun`,
which is a sub-class of :class:`~nitrogen.dfun.DFun`. In fact, an equivalent
way to construct :math:`h(x)` is ``h = CompositeDFun(f,g)``. For multi-variable 
functions, the number of output values of ``g`` (``g.nf``) must equal the number of 
input variables of ``f`` (``f.nx``). The :class:`~nitrogen.dfun.CompositeDFun` attributes
:attr:`~nitrogen.dfun.CompositeDFun.A` and :attr:`~nitrogen.dfun.CompositeDFun.B`
are references to the outer and inner :class:`~nitrogen.dfun.DFun` objects that
define the composition::

    h.A is f  # True
    h.B is g  # True
    

Note that the Python interpreter handles the associativity of ``**`` `right-to-left
<https://docs.python.org/3/reference/expressions.html#the-power-operator>`_, but 
``@`` left-to-right. I.e., ``C ** B ** A`` is evaluated as ``C ** (B ** A)``,
not ``(C ** B) ** A``, and ``A @ B @ C`` is evaluated as ``(A @ B) @ C``, not 
``A @ (B @ C)``. All of these expressions result in the same composite function 
analytically, but they differ in the order with which numerical derivatives are 
handled and combined. This can lead to corresponding differences in performance,
depending on the number of input and output variables of each function. 

Fixed argument functions
~~~~~~~~~~~~~~~~~~~~~~~~

A special sub-class, :class:`~nitrogen.dfun.FixedInputDFun`, is used to 
implement :class:`~nitrogen.dfun.DFun` objects with fixed input arguments,
which is a special case of composition.

Jacobians, Hessians, and optimization
-------------------------------------

Convenience functions for extracting the zeroth derivative (the value), first
derivatives (the gradient or Jacobian), and second derivatives (the Hessian) 
are provided via the :class:`~nitrogen.dfun.DFun` instance methods 
:meth:`~nitrogen.dfun.DFun.val`,
:meth:`~nitrogen.dfun.DFun.jac`, and :meth:`~nitrogen.dfun.DFun.hes`. The return 
value of :meth:`~nitrogen.dfun.DFun.hes` contains the unscaled
derivatives without the permutational pre-factors that are included in the 
complete derivative array returned by :meth:`~nitrogen.dfun.DFun.f`. Continuing
the example from above:

..  doctest:: example-dfun-adf

    >>> df.val(np.array([1.,2.]))
    array([37.])
    >>> df.jac(np.array([1.,2.]))
    array([[ 7., 28.]])
    >>> df.hes(np.array([1.,2.])) # permutational factors not included!
    array([[[ 2.,  0.],
            [ 0., 14.]]])
           
           
Any of  a :class:`~nitrogen.dfun.DFun`'s output functions can be numerically 
optimized via the object's :meth:`~nitrogen.dfun.DFun.optimize` method.

..  doctest:: example-dfun-adf

    >>> xopt, fopt = df.optimize(np.array([0.1234,0.5678]))
    >>> xopt # optimized input arguments
    array([-2.50000024e+00, -7.29258908e-08])
    >>> fopt # optimized output value
    -3.249999999999905
