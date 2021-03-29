"""
nitrogen.tensor
---------------

Tensor networks, contractions, and operators.

"""

import numpy as np 


class Tensor():
    
    """ A general tensor array class, supporting diagonal structure.
    
    .. math::
       
       V_{ijk\\cdots} = v_{abc\\cdots} \delta_{a\cdots} \delta_{b\cdots} \cdots
    
    
    
    Attributes
    ----------
    v : ndarray
        The core array. (The non-zero values.)
    mask : ndarray
        The apparent-to-core index map.
    shape : tuple
        The apparent tensor shape.
    ndim : int
        The order of the tensor. (The number of apparent indices.)
    dtype : data-type
        The data type of the core array.
        
    """
    
    def __init__(self, v, mask = None):
        """
        Parameters
        ----------
        v : ndarray
            The core array for the tensor.
        mask : array_like
            The apparent indices of the tensor. Each axis (0, 1, ...) of 
            the core array `v` must be included at least once. Repeated indices
            form a diagonal set.
        """
        
        self.v = v # The core data array, with reduced shape
        
        if mask is None:
            mask = np.arange(len(self.v.shape))
        #
        # else mask must use 0, 1, ... ndim(v) each at least once
        #
        # 
        if not np.all(np.unique(mask) == np.arange(len(self.v.shape))):
            raise ValueError("invalid mask")
        self.mask = np.array(mask, dtype = np.int32)
        # Determine the apparent shape 
        self.shape = tuple([self.v.shape[i] for i in self.mask])
        self.ndim = len(self.shape) 
        self.dtype = self.v.dtype
        #
        # Each apparent index/tensor-leg belonging to the
        # same axis of the underlying core array has 
        # an joint Kronecker delta implicit, i.e.
        # the apparent tensor is diagonal w.r.t. this
        # set of indices/legs.
        #
    
    def array(self):
        """ Calculate the full array, realizing implicit diagonals
        
        Returns
        -------
        ndarray
            The full apparent tensor.
            
        """         
        nc = np.ndim(self.v) # The number of core axes 
        
        # We will calculate the full array using 
        # einsum. In this notation, the array is
        #
        # V_ijk... = v_abc... I_a... * I_b... * I_c...
        #
        # where v_abc... is the core array.
        # Each axis/index of the core array is contracted
        # with multi-way identity tensor (Kronecker delta)
        # with one index contracted to the core array index
        # and the others corresponding to the apparent
        # index positions defined by the mask.
        
        # The order of each multi-way identity
        # equals the number of times its core index is used 
        # plus 1.
        _,cnts = np.unique(self.mask, return_counts = True)
        In = [eyeN(self.v.shape[i], cnts[i]+1, self.v.dtype) for i in range(nc)]
        # 
        # The array In[i] is the generalized identity tensor to be
        # contracted with core index i
        # 
        # Now we construct the subscript arguments
        # for einsum. The first operand is the core array
        # v. Its subscripts will just be labeled 
        # [0, 1, 2, ... nc-1] 
        # where nc = ndim(v), the number of core axes.
        #
        args = [ self.v, np.arange(nc) ]
        #
        # The remaining operands are the identity tensors
        # for each core index. For each, the first subscript/index
        # is the core axis to be contracted with. The remaining
        # subscripts are the indices of the apparent tensor
        # they correspond to, as defined by their position in 
        # `self.mask`. Because the subscript labels
        # 0, 1, 2, ... nc-1 are already used for the core array,
        # we will label the indices of the apparent tensor
        # with nc, nc+1, nc+2, ... (i.e. the position in the mask
        # plus nc)
        #
        for i in range(nc):
            args.append(In[i])
            w = np.argwhere(self.mask == i)[:,0] + nc 
            args.append([i] + list(w))
        args = tuple(args)
        
        return np.einsum(*args, optimize = True)
    
    def copy(self):
        """ Create a copy of this Tensor
        
        Returns
        -------
        Tensor
            A copy of this Tensor. Both the core and mask ndarrays
            are copied.
            
        """
        return Tensor(self.v.copy(), self.mask.copy())
    
    def __repr__(self):
        return f"Tensor({self.v!r},{self.mask!r})"

class TensorNetwork():
    
    """ A single term tensor network 
    
    Attributes
    ----------
    tensors : list
        A list of Tensor or ndarray objects.
    labels : list
        A list of lists containing the index labels of each
        element of `tensors`. 
    
    """
    
    def __init__(self, tensors, labels):
        """ Create a TensorNetwork
        
        Parameters
        ----------
        tensors : list
            A list of Tensor or ndarray objects.
        labels : list
            A list of lists containing the index labels of each
            element of `tensors`. 
    
        """
        self.tensors = tensors 
        self.labels = labels 
        
        elegs = []
        eshape = []
        for i,t in enumerate(tensors):
            for j,label in enumerate(labels[i]):
                if label < 0 : # An external leg
                    elegs.append(label)
                    eshape.append(t.shape[j])
            
        self.elegs = elegs 
        self.eshape = tuple(eshape)
    
    def con(self, sequence = None, forder = None, check = True):
        """ Contract this network with :func:`~nitrogen.tensor.con`.
        
        Parameters
        ----------
        sequence,forder,check : optional
            See :func:`~nitrogen.tensor.con`.
            
        Returns
        -------
        Tensor
        
        """
        return con(self.tensors, self.labels, sequence = sequence, 
                   forder = forder, check = check)

class TensorOperator():
    """ Base class for tensor operators.
    
    Attributes
    ----------
    shape : tuple
        The one-sided shape of the tensor operator.
    dtype : data-type
        The data type.
    """
    
    def __init__(self, shape, dtype = np.float64):
        self.shape = tuple(shape)
        self.dtype = dtype
        
    
    def contract(self, bra = None, ket = None):
        """
        Contract the tensor operator with tensor networks
        to its left and right.
    
        Parameters
        ----------
        bra : TensorNetwork or None
            The left-hand-side network
        ket : TensorNetwork or None
            The right-hand-side network

        Returns
        -------
        ndarray
            The contracted result
        
        Notes
        -----
        The tensor operator has external indices labeled
        (-1, -2, -3, ...) on each side with dimensions given 
        by `shape`.

        The input tensor networks will be contracted with the 
        operator based on the negative integer elements in 
        their respective `labels` attributes. Missing 
        labels in these networks will not be contracted.

        """
        
        # Handle Nones
        if bra is None:
            bra = TensorNetwork([],[])
        if ket is None:
            ket = TensorNetwork([],[])
        
        # Check external legs 
        for i,label in enumerate(bra.elegs):
            if bra.eshape[i] != self.shape[-1 - label]:
                raise ValueError("External shape of bra does not match "
                                 "operator shape.")
        for i,label in enumerate(ket.elegs):
            if ket.eshape[i] != self.shape[-1 - label]:
                raise ValueError("External shape of ket does not match "
                                 "operator shape.")
        
        return self._contract(bra, ket)
        
    def _contract(self, bra, ket):
        """ bra, ket : TensorNetwork 
            The implementation of contract.
        """
        raise NotImplementedError()
        return 

class ConfigurationOperator():
    """
    Base class for matrix elements in
    a configuration representation.
    
    Attributes
    ----------
    shape : tuple
        The one-sided operator shape
    dtype : data-type
        The operator data type.
        
    """
    def __init__(self, shape, dtype = np.float64):
        self.shape = tuple(shape) 
        self.dtype = dtype 
    
    def block(self, bra_configs, ket_configs = None):
        """
        Calculate a block of the operator matrix.
        
        Parameters
        ----------
        bra_configs : array_like
            A list of left-hand-side configurations.
        ket_configs : array_like or {'symmetric', 'diagonal'}, optional
            A list of right-hand-side configurations. If None,
            then the diagonal block given by `bra_configs` will be 
            calculated.
        """
        
        bra_configs = np.array(bra_configs).reshape((-1, len(self.shape)))
        if ket_configs is None or isinstance(ket_configs, str):
            mode = ket_configs # None or string
            ket_configs = bra_configs
        else:
            mode = None
            ket_configs = np.array(ket_configs).reshape((-1, len(self.shape)))
        
        return self._block(bra_configs, ket_configs, mode)
        
    def _block(self, bra_configs, ket_configs, mode):
        raise NotImplementedError()
        pass # TO BE IMPLEMENTED BY SUB-CLASS 

class DirectSumConfigurationOperator(ConfigurationOperator):
    """ A sum of configuration operators"""
    def __init__(self, *args):
        """ ConfigurationOperator sum """
        n = len(args)
        if n < 1:
            raise ValueError("At least one TensorOperator is required.")
        
        self.shape = args[0].shape 
        self.dtype = args[0].dtype 
        
        for i,A in enumerate(args):
            if A.shape != self.shape:
                raise ValueError(f"The shape of args[{i:d}] does not match.")
            self.dtype = np.result_type(self.dtype, args[i].dtype)
            
        self.terms = args 
    
    def _block(self, bra_configs, ket_configs, mode):
        """ Return the sum of the each term 
        """
        result = 0
        for A in self.terms:
            result += A._block(bra_configs, ket_configs, mode) 
        return result

class SingleIndexOperator(ConfigurationOperator):
    
    """ A 1-group operator in configuration 
    representation. The representation is assumed
    to be separably orthonormal in each configuration index
    """
    
    def __init__(self, A, index, shape):
        """
        Parameters
        ----------
        A : ndarray
            A square matrix representing the 1-body 
            operator for axis `index` of the configuratoin 
            representation.
        index : int
            The 1-body index.
        shape : tuple
            The full configuration space shape
        """
        
        super().__init__(shape, dtype = A.dtype) 
        self.A = A 
        self.index = index 
        
        if index < 0 or index >= len(self.shape):
            raise ValueError("Invalid index ({index:d})")
        
    def _block(self, bra_configs, ket_configs, mode):
        
        nb = bra_configs.shape[0] 
        nk = ket_configs.shape[0] 
        n  = bra_configs.shape[1] # == ket_configs.shape[1], the number of bodies
        
        
        if mode == 'diagonal':
            
            out = np.empty((nb,), dtype = self.dtype)
            
            for i in range(nb):
                bra = bra_configs[i,:] # == ket
                out[i] = self.A[bra[self.index], bra[self.index]]
                
        else:
            
            out = np.empty((nb,nk), dtype = self.dtype) 
            
            for i in range(nb):
                bra = bra_configs[i,:]
                
                for j in range(nk):
                    
                    if j < i and mode == 'symmetric':
                        # If mode is symmetric, then bra and ket list
                        # are the same. Assume upper triangle is equal
                        # to lower triangle 
                        out[i,j] = out[j,i]
                        continue 
                    
                    ket = ket_configs[j,:]
                    
                    # Check that all non-`index` indices are equal
                    # If they are not, then the matrix element
                    # is zero by orthogonality
                    mask = np.arange(n)!=self.index
                    if np.any(bra[mask] != ket[mask]):
                        out[i,j] = 0.0 
                        continue
                    else:
                        # All non-`index` indices are the same. The 
                        # matrix element is equal to the 1-body matrix element
                        #
                        out[i,j] = self.A[bra[self.index], ket[self.index]]
        
        return out 
                
                
        

    
class DirectSumOperator(TensorOperator):
    """
    
    A tensor operator equal to the simple sum of 
    other TensorOperator objects.
    
    """

    def __init__(self, *args):
        """ TensorOperator sum """
        
        n = len(args)
        if n < 1:
            raise ValueError("At least one TensorOperator is required.")
        
        self.shape = args[0].shape 
        self.dtype = args[0].dtype 
        
        for i,T in enumerate(args):
            if T.shape != self.shape:
                raise ValueError(f"The shape of args[{i:d}] does not match.")
            self.dtype = np.result_type(self.dtype, args[i].dtype)
            
        self.terms = args 
    
    def _contract(self, bra, ket):
        """ Contraction implementation.
            Return the sum of the contractions of each term.
        """
        result = 0
        for T in self.terms:
            result += T._contract(bra, ket) 
        return result

class DirectProductOperator(TensorOperator):
    
    """
    Direct product tensor operator, with implicit identities for
    unspecified index-pairs.
    
    Attributes
    ----------
    shape : tuple
        The one-sided shape
    factors : list
        The non-identity tensor factors
    labels : list of lists
        The one-sided labels for each factor
    
    """
    
    def __init__(self, factors, labels, shape):
        """
        A direct product of tensors

        Parameters
        ----------
        factors : list of Tensors
            The factors.
        labels : list of lists
            Each element is a list with the
            negative integer labels that
            the corresponding element represents.
        shape : tuple
            The operator shape. Axes that are
            not referred to in `labels` will 
            implicitly have an Identity tied to 
            them with the correct dimension as given
            by `shape`.
        
        Notes
        -----
        Each label is used once, e.g.
        [-1,-3] means the corresponding tensor
        has shape (n,n,m,m) where `n` is the 
        dimension of label -1 and `m` is the
        dimension of label -3.

        """
        
        nlegs = len(shape) 
        all_labels = sum(labels,[])
        if len(np.unique(all_labels)) != len(all_labels):
            raise ValueError("All labels must be used only once!")
        
        # Determine which labels have Identity factors
        iden_labels = [i for i in range(-1,-nlegs-1,-1) if i not in all_labels]
        
        self.dtype = np.float64
        # Check that the shapes of the factors match 
        # the asserted shape 
        for i,T in enumerate(factors):
            self.dtype = np.result_type(self.dtype, T.dtype)
            for j,label in enumerate(labels[i]):
                s = shape[-label - 1]
                if T.shape[2*j+1] != s or T.shape[2*j+1] != s:
                    raise ValueError(f"The shape of factors[{i:d}] does not"
                                     "match the asserted operator shape.")        
        
        self.shape = shape 
        self.factors = factors 
        self.labels = labels 
        self.iden_labels = iden_labels 
        
    def _contract(self, bra, ket):
        """ The TensorOperator contraction function"""
        #
        # Use con to contract the tensor network
        # 
        # We have up to 3 lists of tensors, whose
        # labels need to be reconciled
        #
        
        #
        # First, merge the bra and ket networks
        #
        # The external indices of the bra and ket need to be mapped
        # to [-1 -2 -3 ... ] -> [-1 -3 -5 ...] for the bra and 
        # to [-1 -2 -3 ... ] -> [-2 -4 -6 ...] for the ket
        #
        # The internal indices of the ket needed to be offset 
        # so as not to conflict with those of the bra
        #
        
        braket_tensors = bra.tensors + ket.tensors
        
        # Calculate the necessary offset for the ket internals.
        # These is the largest positive label in the bra network.
        # If there are no positive labels there, then it should be 0.
        # Use sum(...) to flatten the bra labels and append 0.
        offset = max(sum(bra.labels,[0])) 
        
        bra_labels = [[i          if i > 0 else 2*i+1 for i in lab] for lab in bra.labels]
        ket_labels = [[i + offset if i > 0 else 2*i   for i in lab] for lab in ket.labels]
        
        braket_labels = bra_labels + ket_labels 
        
        neg_labels = [i for g in braket_labels for i in g if i < 0] 
        # Now we merge the bra/ket network with the operator,
        # handling cases involving iden_labels
        #
        #
        op_tensors = [] 
        op_labels = [] 
        dummy_idx = max(sum(braket_labels,[0])) + 1 # keep track of next available dummy label
        for i in self.iden_labels:
            #
            # Tensor axis i is an identity
            #
            # We could simply add an identity tensor
            # to the tensor list and
            # label it appropriately, but if at least one
            # leg is contracted to it, we should 
            # simplify it now
            #
            # Consider 4 cases:
            # 1) Both bra and ket present this axis
            # 2) Only bra presents this axis, no ket
            # 3) Only ket presents this axis, no bra
            # 4) Neither bra nor ket presents this axis
            #
            brai = 2*i + 1 
            keti = 2*i 
            # 1) brai, keti <-- new dummy_idx
            if brai in neg_labels and keti in neg_labels:
                braket_labels = [[dummy_idx if j == brai else j for j in g] for g in braket_labels]
                braket_labels = [[dummy_idx if j == keti else j for j in g] for g in braket_labels]   
                dummy_idx += 1
            # 2) the ket-side external leg is just the bra tensor
            #    brai <-- keti
            elif brai in neg_labels:
                braket_labels = [[keti if j == brai else j for j in g] for g in braket_labels] 
            # 3) converse
            #    keti <-- brai
            elif keti in neg_labels:
                braket_labels = [[brai if j == keti else j for j in g] for g in braket_labels] 
            # 4) an explicit Identity tensor is necessary 
            else: 
                length = self.shape[-i-1] # The dimension of this identity factor 
                v = np.ones((length,))   # A 1-d array of ones
                I = Tensor(v, mask = [0,0]) # A diagonal Tensor identity.
                #
                # Append the operator tensor list and labels list
                op_tensors.append(I)
                op_labels.append([brai,keti])
        #
        # We can now deal with the non-identity factors and their labels
        # Remember, the labels in self.labels only span [-1, ..., -nlegs]
        # Each label really labels an adjacent pair of axes of the corresponding
        # factor (the bra side, then the ket side)
        #
        for i,T in enumerate(self.factors): # For each factor
            op_tensors.append(T)
            
            lab = [] 
            for j,l in enumerate(self.labels[i]): # For each bra/ket pair label
                
                brai = 2*l + 1 # The actual bra-side negative label
                if brai in neg_labels: # A bra leg exists, contract it with the factor
                    braket_labels = [[dummy_idx if k == brai else k for k in g] for g in braket_labels]
                    lab.append(dummy_idx)
                    dummy_idx += 1
                else: # There is no bra for this leg, it remains external instead
                    lab.append(brai)
                
                # Do the same for the ket side
                keti = 2*l
                if keti in neg_labels: # A ket leg exists, contract it
                    braket_labels = [[dummy_idx if k == keti else k for k in g] for g in braket_labels]
                    lab.append(dummy_idx)
                    dummy_idx += 1
                else: # There is no ket for this leg, it remains external instead
                    lab.append(keti) 
                    
            op_labels.append(lab)
        
        # 
        # We now perform the actual contraction 
        # as a full array 
        # 
        ten = braket_tensors + op_tensors 
        lab = braket_labels + op_labels
        
        result_tensor = con(ten, lab)
        result_array = result_tensor.array()
        return result_array

def con(tensors, labels, sequence = None, forder = None, check = True):
    """ An ncon style contraction function. See [NCON]_.
    
    Parameters
    ----------
    tensors : list of Tensors or ndarrays
        The tensors forming the network.
    labels : list of lists of index labels
        The axis of each tensor is labeled with
        an integer index. Positive labels occur
        in pairs and are contracted. Negative
        labels are external legs and uncontracted.
    sequence : list, optional
        The contraction sequence of the positive
        index labels. The default is by ascending order.
    forder : list, optional
        The order of the uncontracted indices in the final
        result tensor. By default, this is in 
        negative ascending order (-1, -2, ...)
    check : boolean, optional
        Perform checks on inputs. The default is True.
    
    Returns
    -------
    Tensor
        Result
        
    Notes
    -----
    Contraction is not necessarily performed strictly in
    the order given by `sequence`. When contracting some
    index labeled by an element of sequence, a look-ahead
    is performed for all later elements in sequence which
    have the same tensor connectivity. These contractions are 
    performed simultaneously and removed from the sequence
    list.
    
    References
    ----------
    .. [NCON] R. N. C. Pfeifer, G. Evenbly, S. Singh, and G. Vidal.
        "NCON: a tensor network contractor for MATLAB". arXiv:1402.0939. (2015)
        https://arxiv.org/abs/1402.0939
    """
    
    ##################################
    # Process `tensors` input
    #
    # If not a list, make it a list
    if not isinstance(tensors, list):
        tensors = [tensors]
    # For each element in the list, if not a Tensor, make a Tensor
    for i in range(len(tensors)):
        if not isinstance(tensors[i], Tensor):
            tensors[i] = Tensor(tensors[i])
    #
    ##################################
    
    ##################################
    # Process `sequence` and `forder` defaults
    if sequence is None:
        sequence = sorted_pos(labels)
    if forder is None: 
        forder = sorted_neg(labels)
    #
    ##################################
    
    ##################################
    # Check inputs
    if check:
        check_labels(tensors, labels, sequence, forder)
    #
    ##################################
    
    ##################################
    # Perform sequential contractions
    #
    while len(sequence) > 0:
        
        # Contracting indices labeled as sequence[0]
        # ------------------------------------------
        # Find which tensor(s) the label connects
        tids = [t for t in range(len(labels)) if sequence[0] in labels[t]]
        # tids has one element if sequence[0] is a trace contraction
        # or two elements if it is a two-tensor contraction
        #
        pairs = [] 
        con_labels = [] 
        #
        # Case 1) A trace within a single tensor 
        if len(tids) == 1: 
            # Find all pairs within labels[tids[0]]
            for i,c in enumerate(labels[tids[0]]):
                # only look in labels[tids[0]] past the position
                # of the current label so as not to double count later
                if c in labels[tids[0]][i+1:]:
                    pairs.append( (i, i + 1 + labels[tids[0]][i+1:].index(c)))
                    con_labels.append(c)
            
            # pairs is a list of axis pairs for tracing
            # con_labels is the list of labels for these pairs 
            # Calculate the new tensor and its new list of labels
            new_T = tensorTrace(tensors[tids[0]], pairs)
            new_label = [i for i in labels[tids[0]] if i not in con_labels]
            
        #
        # Case 2) Contractions between two different tensors
        #
        elif len(tids) == 2:
            # Find all pairs overlapping labels[tids[0]] and
            # labels[tids[1]]
            for i,c in enumerate(labels[tids[0]]):
                if c in labels[tids[1]]:
                    pairs.append( (i, labels[tids[1]].index(c) ))
                    con_labels.append(c) 
            #
            # pairs is a list of axis pairs for contracting
            # pairs[i][0] is an axis of tensors[tids[0]] and
            # pairs[i][1] is an axis of tensors[tids[1]] 
            # con_labels is the list of labels for these pairs 
            A,B = tensors[tids[0]], tensors[tids[1]]
            new_T = tensorContract(A,B,pairs)
            new_label = [i for i in labels[tids[0]] if i not in con_labels] + \
                [i for i in labels[tids[1]] if i not in con_labels]
            
        else:
            raise ValueError("A contraction index was connected to more than"
                             "two tensors")
          
        #
        # Remove the old tensor(s) and labels and append the 
        # new tensor and labels
        #
        for t in sorted(tids, reverse=True):
            del tensors[t] 
            del labels[t] 
        tensors.append(new_T)
        labels.append(new_label)
        # 
        # Remove all contracted labels from the sequence list 
        sequence = [i for i in sequence if i not in con_labels]
    #
    ##################################
    
    ##################################
    # After contractions are complete,
    # any remaining tensors form
    # a direct product
    #
    T = tensorDirectProduct(tensors, force_copy = False) # Direct product of remaining tensors
    labels = sum(labels,[]) # Flattened labels list, this should be unique negative integers
    perm = [labels.index(idx) for idx in forder]
    
    return tensorPermute(T, perm)
    
        
    
def sorted_pos(labels):
    """ Return unique positive labels in ascending order"""
    idx = sum(labels, []) 
    pos = [i for i in idx if i > 0]
    return list(np.sort(np.unique(pos)))

def sorted_neg(labels):
    """ Return unique negative labels in descending order"""
    idx = sum(labels, [])
    neg = [i for i in idx if i < 0]
    return list(-np.sort(-np.unique(neg)))
        
def check_labels(tensors, labels, sequence, forder):
    """ Check labels, sequence and forder for a given
    tensor list """
    
    # Checks
    # ------
    # 1) len(tensors) = len(labels)
    # 2) ndim of tensors[i] = len(labels[i])
    # 3) Each element in sequence is positive and occurs twice in labels
    #
    # 4) Each element in forder is negative and occurs once in labels
    # 5) Each element in labels occurs somewhere in sequence or forder
    #
    
    # 1) 
    if len(tensors) != len(labels):
        raise ValueError("The number of tensors {:d} does not equal the"
                         "number of index label lists {:d}".format(len(tensors), len(labels)))
    
    # 2)
    for i in range(len(tensors)):
        if tensors[i].ndim != len(labels[i]):
            raise ValueError("tensors[{:d}] has {:d} dimension(s) but "
                             "labels[{:d}] has {:d} index label(s)".format(i,tensors[i].ndim,
                                                                             i, len(labels[i])))
    #
    all_labels = sum(labels,[]) 
    # 3)
    for idx in sequence:
        if all_labels.count(idx) != 2:
            raise ValueError(f"Index label {idx:d} must occur twice in labels")
        if idx <= 0:
            raise ValueError("Contracted index labels must be positive.")
        # We could also check that contracted labels have equal tensor dimension
        # but this will throw an error in the actual contraction routines,
        # so we'll skip it for now 
    # 4)
    for idx in forder:
        if all_labels.count(idx) != 1:
            raise ValueError(f"Index label {idx:d} must occur once in labels")
        if idx >= 0:
            raise ValueError("Uncontracted index labels must be negative.")
    # 5)
    for idx in all_labels:
        if idx > 0 and sequence.count(idx) != 1:
            raise ValueError(f"Index label {idx:d} must occur once in sequence")
        if idx < 0 and forder.count(idx) != 1:
            raise ValueError(f"Index label {idx:d} must occur once in forder")
            
    return
            
def eyeN(d,N, dtype = np.float64):
    """ Construct a d x d x ... (N times)
        identity tensor
    
    Parameters
    ----------
    d : int
        The dimension of each index
    N : int
        The number of indices.
    dtype : data-type, optional
        The data-type. The default is np.float64.
    
    Returns
    -------
    I : ndarray
        The N-dimensional identity tensors
        
    Notes
    -----
    If `N` == 1, then a 'ones' array of length `d`
    is returned.
    
    """
    
    shape = (d,) * N 
    if N > 1:
        I = np.zeros(shape, dtype = dtype)
        np.fill_diagonal(I, 1) 
    else:
        # N == 1
        # A single index delta tensor is
        # just all ones.
        I = np.ones(shape, dtype = dtype)
    return I

def mask2train(mask):
    """ Calculate the delta train
    core and apparent indices from a mask
    
    Parameters
    ----------
    mask : ndarray
        Tensor mask
        
    Returns
    -------
    train : nested list
        A list of delta factors with core and apparent
        indices grouped separately.
        
    """
    
    mask = np.array(mask)
    nc = len(np.unique(mask)) # The number of core indices 
    #na = len(mask) # The number of apparent indices
    
    train = []
    for i in range(nc): # For each core index
        # Find which apparent indices map to 
        # this core index
        a_ind = list(np.argwhere(mask == i)[:,0])
        train.append( [ [i], a_ind ] )
        
    return train 

def reduceTrain(mask, pairs):
    
    """
    Compute a reduced train of delta factors by contracting
    pairs of indices.
    
    Parameters
    ----------
    mask : ndarray
        Mask array 
    
    Returns
    -------
    train : list
        Train of new delta factors
    app2app : ndarray
        Map from left-over old apparent indices to new apparent indices
        
    """
    
    #
    # The train is a nested list. 
    #
    # [ [ [...], [...] ],
    #   [ [...], [...] ],
    #        ....
    #]
    # 
    # Each element is a set of two lists and corresponds to a single 
    # delta tensor. This delta tensor has some set of core index labels
    # (0, 1, ..., nc-1) given by the first sub-list and and another set of 
    # apparent index labels (0, 1, ..., na-1) given by the second
    # sub-list
    #

    # 
    #
    # For each pairs of indices, we are contracted (tracing) over the current
    # tensors apparent indices
    #
    # If the pair of apparent indices occurs in the same delta factor
    # in the train, then they are simply removed from that factor
    # If they occur in different factors, these factors are 
    # merged and then the pair is removed
    #
    # To keep track of which factor each apparent index is current in,
    # we will use a "train id" list, called tid. Before any trace contractions
    # have been processed, the train is the just the raw train defined by 
    # the original mask, the train id's are determined directly by the mask,
    # and the original apparent indices just map to themselves as new
    # apparent indices. 
    #
    # Initialize tid, train, and app2app
    tid = mask.copy()
    train = mask2train(mask)
    app2app = np.arange(len(mask)) # Map between original and new apparent indices
    
    
    for pair in pairs:
        i = pair[0] # Apparent index i
        j = pair[1] # Apparent index j
        ti = tid[i] # The train i is in
        tj = tid[j] # The train j is in 
        if ti == tj: # i and j are in the same delta factor
            # Simply remove both apparent indices from the factor
            # and update the old-to-new apparent index map
            train[ti][1].remove(i); app2app[i:] -= 1
            train[ti][1].remove(j); app2app[j:] -= 1
        else: # i and j are on different delta factors 
            # combine these factors
            # by merging the later one with the earlier one
            # At the same time, remove i and j from their respective factors
            # and update app2app map
            train[min(ti,tj)][0] = train[ti][0] + train[tj][0]
            train[min(ti,tj)][1] = train[ti][1] + train[tj][1]
            train[min(ti,tj)][1].remove(i); app2app[i:] -= 1
            train[min(ti,tj)][1].remove(j); app2app[j:] -= 1
            # Get rid of the later factor
            del train[max(ti,tj)]
            # and update the train id:
            tid[tid == max(ti,tj)] = min(ti,tj) # App. indices that got moved 
            tid[tid > max(ti,tj)] -= 1  # App. indices that got shifted one up in the train
            # note that tid[i] only remains valid for
            # apparent indices `i` that haven't been traced over
            # yet
        
    return train, app2app 
    
def train2subs_mask(train,app2app, nc_old, na_new):
    """
    Compute the einsum subscripts for core processing 
    as well as the new Tensor mask given a 
    reduced, contracted train of delta factors.
    
    Parameters
    ----------
    train : list
        the reduced train, as returned by reduceTrain()
    app2app : ndarray
        the apparent index map, as returned by reduceTrain()
    nc_old : int
        The original number of core indices
    na_new : int
        The new number of apparent indices 
    
    Returns
    -------
    core_sub, out_sub : list
        einsum subscripts for core processing
    new_mask : list
        New Tensor mask.
    """
    
    # Initialize core_sub, new_maks and out_sub
    core_sub = [None for i in range(nc_old) ] # core subscript for einsum
    new_mask = [None for i in range(na_new) ] # new mask
    out_sub = [] # output subscripts for einsum, initialize to nothing
    
    dummy_idx = 0 # keep track of einsum index 
    new_core_idx = 0 # keep track of new core index (for purposes of new mask)
    
    for i in range(len(train)): # For each factor in the train
        # The old core indices in this factor share a common
        # dummy index in the einsum
        for c in train[i][0]:
            core_sub[c] = dummy_idx 
        #
        # If there are also apparent indices in this factor, then
        # this same dummy index appears on the output subscripts
        # of the einsum as a new core index.
        # The leftover apparent indices have positions in
        # the new mask tied to this new core index.
        #
        if len(train[i][1]) > 0: 
            out_sub.append(dummy_idx) # Place core index in einsum output
            
            for a in train[i][1]:
                # Update the new mask with remaining apparent indices
                new_mask[app2app[a]] = new_core_idx 
            
            new_core_idx += 1
        
        # Increment dummy index 
        dummy_idx += 1 
    
    return core_sub, out_sub, new_mask 
    

def tensorTrace(A,axes):
    """
    Perform a multi-index trace over
    sets of pairs of indices.

    Parameters
    ----------
    A : Tensor or ndarray
        The input tensor.
    axes : list of (2,)
        The axis pairs to trace over. Axes must
        all be unique and be between 0 and A.ndim-1.

    Returns
    -------
    B : Tensor
        The result

    """
    #
    # Convert A to Tensor if necessary
    if not isinstance(A, Tensor):
        A = Tensor(A)
    # Reduce the delta factor train given the
    # pairs of traced axes
    #
    train, app2app = reduceTrain(A.mask, axes)
    #
    # The result tensor is currently equal 
    # the the original core tensor einsummed with 
    # the train of delta factors. We now 
    # compute the equivalent new core tensor
    # and mask in standard diagonal-tensor format
    #
    
    #
    # Using the reduced train, computing the einsum
    # subscript indices for the original core, the core output
    # and the new Tensor mask
    #
    nc_old = np.ndim(A.v)  # The number of old core indices
    na_new = len(A.mask) - 2 * len(axes) # The number of new apparent indices
    core_sub, out_sub, new_mask = train2subs_mask(train, app2app, nc_old, na_new)
    
    # Compute the new core
    new_core = np.einsum(A.v, core_sub, out_sub, optimize = True)
    
    # Return a new Tensor with the new core and new mask
    #
    return Tensor(new_core, new_mask) 
            
        
def tensorContract(A,B,axes):
    """
    Contract diagonal tensors

    Parameters
    ----------
    A, B : Tensor or ndarray
        Tensor to be contracted.
    axes : list of (2,)
        Apparent axis pairs to be contracted. The first element
        of each pair is an axis of `A`. The second element
        is an axis of `B`.

    Returns
    -------
    C : Tensor
        The result 

    """
    
    # Contract two tensors is the same as
    # first forming their direct product
    # and then doing a trace
    #
     
    # Convert A and B to Tensors if necessary
    if not isinstance(A, Tensor):
        A = Tensor(A)
    if not isinstance(B, Tensor):
        B = Tensor(B)
    
    # Calculate the effective axes and mask for
    # the direct product of A and B
    # For axes, the second element of each pair (which refers to B)
    # should be offset by the number of apparent axes of A
    # For the mask, we concatenate the two masks of A and B 
    # and offset that of B by the number of core axes of A
    #
    axes_offset = [(pair[0], pair[1] + len(A.mask)) for pair in axes]
    mask_concat = np.concatenate( (A.mask.copy(), B.mask.copy() + np.ndim(A.v)))
    train, app2app = reduceTrain(mask_concat, axes_offset)
    
    ncA = np.ndim(A.v)  # The number of A core indices
    ncB = np.ndim(B.v)  # The number of B core indices
    naA = len(A.mask)   # The number of A apparent indices
    naB = len(B.mask)   # The number of B apparent indices 
    
    #
    # Compute the einsum subscripts and new mask
    #
    nc_old = ncA + ncB # The original number of core indices
    na_new = naA + naB - 2*len(axes) # The final number of apparent indices
    core_sub, out_sub, new_mask = train2subs_mask(train, app2app, nc_old, na_new)
    
    #
    # Compute the new core
    #
    new_core = np.einsum(A.v, core_sub[:ncA],
                         B.v, core_sub[ncA:], 
                         out_sub, optimize = True )
    #
    # Return a new Tensor with the new core and new mask
    #
    return Tensor(new_core, new_mask) 
    
def tensorDirectProduct(tensors, force_copy = False):
    """ Direct product of multiple tensors. 
        This may share references with input tensors unless force_copy is True
    """
    
    if len(tensors) == 0:
        raise ValueError("tensors must contain at least one element")
    
    for i in range(len(tensors)):
        if not isinstance(tensors[i], Tensor):
            tensors[i] = Tensor(tensors[i])
    
    
    a = tensors[0].v # The first core
    a_mask = tensors[0].mask 
    for i in range(1,len(tensors)):
        b = tensors[i].v # A new core for direct product
        b_mask = tensors[i].mask
        
        na = len(a.shape)
        nb = len(b.shape)
        
        # Calculate direct product of cores 
        a = np.einsum(a, np.arange(na),
                      b, np.arange(nb) + na,
                      np.arange(na+nb), optimize=True)
        # Calculate new mask
        a_mask = np.concatenate( (a_mask, b_mask + na))
    
    if force_copy:
        a = a.copy()
        a_mask = a_mask.copy()
    
    return Tensor(a,a_mask)
        
def tensorPermute(A, perm, copy_core = False):
    """ Permute Tensor axes """
    
    new_mask = [A.mask[i] for i in perm]
    v = A.v 
    if copy_core:
        v = v.copy() 
        
    return Tensor(v, new_mask) 
    

def diagonal(a):
    """ Return the Tensor with `a` as its simple diagonal.
    
    .. math::
     
       A_{ii'jj'\\cdots} = a_{ij\\cdots}\\delta_{ii'}\\delta_{jj'} \\cdots
    
    Parameters
    ----------
    a : ndarray
        The core array
    
    Returns
    -------
    A : Tensor
        A diagonal tensor with `a` along its generalized diagonal.
        
    """
    
    n = len(a.shape) # The number of axes
    mask = np.repeat(np.arange(n), [2 for i in range(n)])
    return Tensor(a,mask)
        

def label2grppos(labels):
    """
    Convert a list of lists of labels to 
    a group/position map 
    """
    
    ngroups = len(labels)
    
    all_labels = sum(labels,[])
    nlabels = max(all_labels) + 1 
    
    if list(np.arange(nlabels)) != list(np.unique(all_labels)):
        raise ValueError(f"Invalid labels: {labels!r}")
        
    grp = []
    pos = [] 
    for i in range(nlabels):
        # Find where label `i` is 
        for g in range(ngroups):
            if i in labels[g]:
                # Found the group
                grp.append(g)
                pos.append(labels[g].index(i))
                break 
    
    return grp,pos 
    
    