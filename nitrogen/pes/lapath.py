"""
lapath.py

Least-action paths

"""

import numpy as np 
import nitrogen
import matplotlib.pyplot as plt 

def calcPathAction(qpath, V, cs, masses, band_action = 1000.0,
                    kinetic_energy = 0.0, deriv = 0,
                    Vmin = 0, ignore_g_deriv = False):
    """
    Calculate the path action and its derivatives.

    Parameters
    ----------
    qpath : (nq,N) ndarray
        The coordinate path of `nq` coordinates along `N` nodes.
    V : DFun
        The potential energy surface supporting `deriv >= 1`.
    cs : CoordSys
        The coordinate system
    masses : array_like
        The masses.
    band_action : float, optional
        The band action force constant. The default is 1000.0.
        This parameter has units of [action] / [arc length]^2 
    kinetic_energy : float, optional
        An energy offset added to the potential. The default is 0.
        This parameter has units of [energy].
    deriv : {0,1,2}, optional
        The derivative order to calculate. The default is 0.
    Vmin : float, optional
        The minimum energy to subtract. The default is 0.
    ignore_g_deriv : bool, optional
        If True, derivatives of the metric tensor are ignored. 
        The default is False.

    Returns
    -------
    
    s : float
        The path arc length.
    I : float
        The path action.
    B : float
        The elastic band action.
        
    If `deriv` >= 1:
    
    Ds : (nq,N) ndarray
        The derivative of `s` with respect to the path node parameters.
    DI : (nq,N) ndarray
        The derivative of `I` with respect to the path node parameters.
    DB : (nq,N) ndarray
        The derivative of `B` with respect to the path node parameters. 
        
    If `deriv` >= 2:
        
    D2s : (nq,N,nq,N) ndarray
        The Hessian of `s` with respect to the path node parameters.
    D2I : (nq,N,nq,N) ndarray
        The Hessian of `I` with respect to the path node parameters.
    D2B : (nq,N,nq,N) ndarray
        The Hessian of `B` with respect to the path node parameters. 
        
        
    Notes
    -----
    
    The path action is approximated by a sum over sequential linear segments
    between nodes in coordinate space.
    
    The length of each path segment is calculated as 
    
    ..  math ::
        
        \\delta s_i = \\sqrt{ \\delta q_i \\frac{\\tilde{g}_i + \\tilde{g}_{i+1}}{2} \\delta q_i },
        
    where :math:`\\delta q_i = (q_{i+1} - q_i)` is the path segment displacement
    and :math:`\\tilde{g}` is the effective
    vibrational metric tensor (i.e. the inverse of the vibrational block of the full inverse
    metric tensor :math:`G`.)
    
    The effective momentum at each node is :math:`p_i = \\sqrt{2(V(q_i) + E)}`, where :math:`E` is
    the kinetic energy offset. The total action is then
    
    ..  math ::
        
        I = \\sum_i \\frac{p_i + p_{i+1}}{2} \\delta s_i
    
    The arc length and action units are defined by the `V`, `cs`, and `masses` parameters.
    
    
    An elastic band action is also calculated as 
    
    ..  math ::
        
        B = \\sum_i b (\\delta s_i - \\delta s_{i+1})^2,
        
    where `b` is the band action force constant.
    
    Note that the minimum energy of `V` will be subtracted before computing the
    action and associated derivatives.
    
    This implementation ignores the kinetic pseudo-potential contribution, which may
    become significant if the path approaches singular points of the metric tensor.

    
    """
    
    # 
    # qpath ... (nq,N) the path nodes 
    #
    # Returns
    # 
    # I ... the path action 
    # s ... the path length
    # B ... the band action 
    # 
    # and derivatives as requested by `deriv`.
    #
    
    nq,N = qpath.shape 
    KE = kinetic_energy # the kinetic offset energy 

    ###################################
    # Prepare the potential and coordinate system
    # quantities
    #
    if deriv > 2:
        raise ValueError("Only deriv = 0, 1, or 2 is supported.")
        
    #
    # Calculate the potential energy and its derivatives 
    #
    if deriv == 0:
        V = V.val(qpath)[0] 
    elif deriv == 1:
        vj = V.vj(qpath)
        V = vj[0][0]
        DV = vj[1][0] 
    elif deriv == 2:
        vjh = V.vjh(qpath) 
        V = vjh[0][0]       # (N,)
        DV = vjh[1][0]      # (nq,N)
        D2V = vjh[2][0]     # (nq,nq,N)
    
    V -= Vmin # Remove minimum energy 
    
    # Calculate the metric tensor and its derivatives 
    if not ignore_g_deriv:
        dg = cs.Q2g(qpath, masses = masses, deriv = deriv) # (nd, npacked, N)
        dG,_ = nitrogen.dfun.sym2invdet(dg, deriv, nq)
        dgvib,_ = nitrogen.dfun.sym2invdet(dG[:, :(nq*(nq+1))//2],
                                         deriv, nq) # (nd, npacked, N)
    else:
        # Calculate only the value and set derivatives to zero 
        dg_val = dg = cs.Q2g(qpath, masses = masses, deriv = 0) # (nd = 1, npacked, N)
        dG_val,_ = nitrogen.dfun.sym2invdet(dg_val, 0, nq)
        dgvib_val,_ = nitrogen.dfun.sym2invdet(dG_val[:, :(nq*(nq+1))//2],
                                         0, nq) # (nd, npacked, N)
        nd = nitrogen.dfun.nderiv(deriv, nq)
        dgvib = np.zeros_like(dgvib_val, shape = (nd,) + dgvib_val.shape[1:] )
        np.copyto(dgvib[0], dgvib_val[0])
    
    
    if deriv >= 0:    
        gv = np.zeros((N,nq,nq))
        
        idx1 = 0 
        for i in range(nq):
            for j in range(i+1):
                gv[:,i,j] = dgvib[0, idx1, :]
                gv[:,j,i] = dgvib[0, idx1, :] 
                idx1 += 1
    
    if deriv >= 1:
        Dgv = np.zeros((nq,N,nq,nq))
        idx1 = 0 
        for i in range(nq):
            for j in range(i+1):
                Dgv[:,:,i,j] = dgvib[1:(nq+1), idx1, :]
                Dgv[:,:,j,i] = dgvib[1:(nq+1), idx1, :] 
                idx1 += 1
        
    if deriv >= 2:
        D2gv = np.zeros((nq,nq,N,nq,nq)) # derivative m,n / node : / tensor element i,j
        
        idx1 = 0 # i,j are the packed symmetric matrix index. lower triangle row major
        for i in range(nq):
            for j in range(i+1):
                
                idx2 = 0  # m,n are the derivative indices. upper triangle row major
                for m in range(nq):
                    for n in range(m,nq):
                        
                        if m == n: 
                            val = 2 * dgvib[1+nq+idx2, idx1, :]
                        else: 
                            val = dgvib[1+nq+idx2, idx1, :]
                        D2gv[m,n,:,i,j] = val
                        D2gv[m,n,:,j,i] = val
                        D2gv[n,m,:,i,j] = val
                        D2gv[n,m,:,j,i] = val
                        
                        idx2 += 1 
                        
                idx1 += 1
    
    

    # Summary of path quantities 
    # 
    # (derivative indices first, then path index, then tensor indices)
    #
    #
    # V   ...       (N,)        The potential energy 
    # DV  ...    (nq,N)         The potential energy gradient 
    # D2V ... (nq,nq,N)         The potential energy Hessian 
    #
    # gv  ...       (N,nq,nq)   The vibrational metric tensor
    # Dgv ...    (nq,N,nq,nq)   The first derivatives of gv 
    # D2gv .. (nq,nq,N,nq,nq)   The second derivatives of gv 
    #

    # Calculate the path length and its derivatives
    # w.r.t node positions
    #
    # The path length is calculated by the trapezoid rule
    # along each node-to-node segment 
    # 
    # Keep a record of the length of each segment 
    #
    # Note that we use little `d` here to label 
    # quantities for single segments
    #
    ds = _calc_path_length_trap(qpath, gv) # (N-1,)
    s = sum(ds)
    
    if deriv >= 1:
        # The derivatives of each segment are non-zero only
        # with the before `b` and after `a` nodes.
        # 
        # Calculate the first derivatives of ds and ds**2
        # with respect to the nodes before and after 
        Dds_b, Dds_a = _calc_path_length_grad(qpath, gv, Dgv, ds)
    
        # Compute derivatives of total path length 
        Ds = np.zeros((nq,N))
        
        for i in range(N-1):
            Ds[:,i] += Dds_b[:,i]   # Derivatives of segment i from node before 
            Ds[:,i+1] += Dds_a[:,i] # Derivatives of segment i from node after
    
    if deriv >= 2:
        # Calculate the second derivatives of ds and ds**2
        # with respect to the nodes before and after 
        D2ds_bb, D2ds_ba, D2ds_aa = _calc_path_length_hes(qpath, gv, Dgv, D2gv, ds, Dds_b, Dds_a)
    
        # Compute derivatives of total path length 
        D2s = np.zeros((nq,N,nq,N)) 
        #
        # Proceed segment by segment
        for i in range(N-1):
            #
            # The i**th segment
            # The before node is `i`
            # The after node is `i+1`.
            #
            D2s[:,i,  :,i  ] += D2ds_bb[:,:,i]
            D2s[:,i+1,:,i+1] += D2ds_aa[:,:,i] 
            D2s[:,i,  :,i+1] += D2ds_ba[:,:,i] 
            D2s[:,i+1,:,i  ] += (D2ds_ba[:,:,i]).T 
        
    ################################
    # We now move onto the action integral.
    # This action will be computed segment-by-segment
    # using a trapezoid rule.
    #
    # Each segment needs the integral of 
    # 
    #  sqrt[2*(V + E)] * ds 
    # 
    # The trapezoid rule will be used by averaging
    # the values of the sqrt[...], i.e. the momentum,
    # not by taking the sqrt of the average potential.
    # 
    # (In the high N limit, these should converge.)
    # 

    
    #
    # Before continuing, we first calculate the value and
    # derivatives of the momentum at each node w.r.t.
    # the node position.
    #
    p = np.zeros((N,))
    for i in range(N):
        # p = sqrt[2 * (V + KE)]
        # Avoid divide-by-zero error by adding machine eps
        #
        p[i] = np.sqrt(2 * (V[i] + KE + 1e-15))
    
    if deriv >= 1:
        
        # Gradient of momentum
        
        Dp = np.zeros((nq,N))
        for i in range(N):
            #
            # Dp = 0.5/p * (2 * DV) 
            #    = DV / p 
            #
            Dp[:,i] = DV[:,i] / (p[i] + 1e-15)
    
    if deriv >= 2: 
        
        # Hessian of momentum 
        
        D2p = np.zeros((nq,nq,N)) 
        for i in range(N):
            #
            #  DaDb[p] = (DaDb[V] - Da[p] Db[p] ) / p
            #
            D2p[:,:,i] = (D2V[:,:,i] - np.outer(Dp[:,i],Dp[:,i])) / (p[i] + 1e-15)
    #
    ################################
    
    ################################
    # Compute the action integral
    #
    # Each path segment contributes
    # the mean momentum times the segment length
    #
    # 
    dI = np.zeros((N-1,))  # the action of each segment 
    for i in range(N-1):
        # dI[i] 
        # Action between node i and i + 1 
        pbar = 0.5 * (p[i] + p[i+1])
        dI[i] = pbar * ds[i]  
    I = sum(dI) 
    
    if deriv >= 1:    
        # Compute the derivatives of the action of 
        # each segment.
        DdI_b = np.zeros((nq,N-1)) # The derivative of segment i w.r.t. node before
        DdI_a = np.zeros((nq,N-1)) # The derivative of segment i w.r.t. node after 
        
        for i in range(N-1):
            # For segment `i` 
            pbar = 0.5 * (p[i] + p[i+1])
            DdI_b[:,i] = Dp[:,i] * ds[i] / 2 + pbar * Dds_b[:,i] 
            DdI_a[:,i] = Dp[:,i+1]*ds[i] / 2 + pbar * Dds_a[:,i] 
        
        # Compute derivatives of total path action 
        DI = np.zeros((nq,N)) 
        for i in range(N-1):
            DI[:,i] += DdI_b[:,i]
            DI[:,i+1] += DdI_a[:,i]
        
    if deriv >= 2:
        # 
        # Compute the second derivatives of the 
        # action of each segment. 
        #
        D2dI_bb = np.zeros((nq,nq,N-1)) # The before/before block 
        D2dI_ba = np.zeros((nq,nq,N-1)) # The before/after block 
        D2dI_aa = np.zeros((nq,nq,N-1)) # The after/after block 
        
        for i in range(N-1):
            # segment i 
            pbar = (p[i] + p[i+1]) / 2 
            
            # before/before block 
            D2dI_bb[:,:,i] += D2p[:,:,i] * ds[i] / 2 
            D2dI_bb[:,:,i] += pbar * D2ds_bb[:,:,i]
            D2dI_bb[:,:,i] += np.outer(Dp[:,i], Dds_b[:,i]) / 2 
            D2dI_bb[:,:,i] += np.outer(Dds_b[:,i], Dp[:,i]) / 2
        
            # after/after block 
            D2dI_aa[:,:,i] += D2p[:,:,i+1] * ds[i] / 2 
            D2dI_aa[:,:,i] += pbar * D2ds_aa[:,:,i]
            D2dI_aa[:,:,i] += np.outer(Dp[:,i+1], Dds_a[:,i]) / 2 
            D2dI_aa[:,:,i] += np.outer(Dds_a[:,i], Dp[:,i+1]) / 2
            
            # before/after block 
            D2dI_ba[:,:,i] += np.outer(Dds_b[:,i], Dp[:,i+1])/2 
            D2dI_ba[:,:,i] += np.outer(Dp[:,i], Dds_a[:,i])/2 
            D2dI_ba[:,:,i] += pbar * D2ds_ba[:,:,i] 
        
        # Compute total derivatives
        D2I = np.zeros((nq,N,nq,N)) 
        
        for i in range(N-1):
            #
            # The i**th segment
            # The before node is `i`
            # The after node is `i+1`.
            #
            D2I[:,i,  :,i  ] += D2dI_bb[:,:,i]
            D2I[:,i+1,:,i+1] += D2dI_aa[:,:,i] 
            D2I[:,i,  :,i+1] += D2dI_ba[:,:,i] 
            D2I[:,i+1,:,i  ] += (D2dI_ba[:,:,i]).T 

    
    #
    # Calculate the node distribution potential,
    # which keeps the nodes approximately uniformly
    # spaced. (The `potential` is really an action because
    # it is expected to be added to the path action.)
    #
    # Each pair of sequential segments contributes
    # to the potential as 
    #
    #  k * (ds[i] - ds[i+1])**2
    #
    # where `k` has units of [action] / [arc length]**2 
    # 
    B = 0.0 
    k = band_action 
    for i in range(N-2):
        #
        # Segments i -- i+1 and i+1 -- i+2
        B += k * (ds[i] - ds[i+1])**2 
    
    if deriv >= 1: 
        
        #
        # Derivatives of the band action w.r.t
        # each node position 
        # 
        DB = np.zeros((nq,N))
        for i in range(N-2):
            #
            # Contributions from the band action 
            # between segments i==i+1 and i+1==i+2
            #
            #     B <-- k * (ds[i] - ds[i+1])**2 
            #  D[B] <-- 2 * k * (ds[i] - ds[i+1]) * D[ ds[i] - ds[i+1] ]
            #
            # There are three nodes (i, i+1, and i+2)
            # that can affect this action 
            #
            
            pre = 2 * k * (ds[i] - ds[i+1]) # the pre-factor
            
            # First node, i, only contributes via ds[i], which is the 
            # the segment *after* i.
            #
            DB[:,i] += pre * Dds_b[:,i] 
            
            # Last node, i+2, only contributes via ds[i+1], which is the 
            # the segment *before* i+2.
            # Note negative sign 
            #
            DB[:,i+2] += -pre * Dds_a[:,i+1]
            
            # The middle node, i+1, contributes via both ds[i] and ds[i+1]
            #
            DB[:,i+1] += pre * Dds_a[:,i] 
            DB[:,i+1] += -pre * Dds_b[:,i+1]
    
    if deriv >= 2:
        #
        # Second derivatives of the band action
        #
        D2B = np.zeros((nq,N,nq,N)) 
        for i in range(N-2):
            #
            # For each band segment pair, there are 
            # three types of nodes: before (b), middle (m),
            # and after (a).
            # This leads to six types of second derivatives:
            # bb, mm, aa, bm, ba, ma 
            
            delta = ds[i] - ds[i+1]
            
            #
            # before/before
            D2B[:,i,:,i] += 2*k*(np.outer(Dds_b[:,i], Dds_b[:,i]) + delta * D2ds_bb[:,:,i])
            
            # after/after 
            D2B[:,i+2,:,i+2] += +2*k*(+np.outer(Dds_a[:,i+1], Dds_a[:,i+1]) +
                                      -delta * D2ds_aa[:,:,i+1]) 
            
            # middle/middle 
            v1 = Dds_a[:,i] - Dds_b[:,i+1] 
            D2B[:,i+1,:,i+1] += 2*k * (np.outer(v1,v1) + delta * (D2ds_aa[:,:,i] - D2ds_bb[:,:,i+1])) 
            
            # before/after 
            hba = -2*k*np.outer(Dds_b[:,i], Dds_a[:,i+1]) 
            D2B[:,i,:,i+2] += hba 
            D2B[:,i+2,:,i] += hba.T 
            
            # before/middle 
            hbm = 2*k*(np.outer(Dds_b[:,i], v1) + delta * D2ds_ba[:,:,i]) 
            D2B[:,i,:,i+1] += hbm 
            D2B[:,i+1,:,i] += hbm.T 
            
            # middle/after 
            hma = -2*k*(np.outer(v1,Dds_a[:,i+1]) + delta * D2ds_ba[:,:,i+1])
            D2B[:,i+1,:,i+2] += hma 
            D2B[:,i+2,:,i+1] += hma.T 
    #
    # Done!
    #
    ############################    
    
    if deriv == 0:
        
        return s, I, B 
    
    elif deriv == 1: 
        
        return s, I, B, \
               Ds, DI, DB 
    
    elif deriv == 2:
        
        return s, I, B, \
               Ds, DI, DB, \
               D2s, D2I, D2B 
    

def _calc_gvib(dg):
    """
    Calculate the effective vibrational metric (for zero total 
    angular momentum) and its first derivatives.    
    
    Parameters
    ----------
    dg : (nd, (nv+3)*(nv+4)/2, ...)
        The full rovibrational metric tensor and its first derivatives in packed
        format.
        
    Returns
    -------
    None.

    """
    
    # Calculate the metric tensor and its derivatives
    dg = np.moveaxis(dg, 1, 0)  # Move the packed index to the front 
    dg = nitrogen.linalg.packed.symfull(dg) # Unpack to (nq+3,nq+3,nd,...)
    dg = np.moveaxis(dg, [0,1], [-2,-1]) # Move to (nd,...,nq+3,nq+3) 
    
    # Calculate the inverse and its derivatives 
    dG = np.empty_like(dg) 
    dG[0] = np.linalg.inv(dg[0]) # Value 
    nq = dg.shape[-1] - 3 # The number of coordinates
    for i in range(nq):
        dG[i+1] = -dG[0] @ dg[i+1] @ dG[0] # Derivative 
    
    # Extract the vibrational block of G, then
    # re-invert it 
    #
    dgv = np.empty(shape = dg.shape[:-2] + (nq,nq))
    dgv[0] = np.linalg.inv(dG[0,...,:nq,:nq])
    for i in range(nq):
        dgv[i+1] = -dgv[0] @ dG[i+1,...,:nq,:nq] @ dgv[0] # Derivative 
    
    return dgv 

def _calc_path_length_trap(qpath, gvib):
    """
    Calculate a path arc length using trapezoid
    rule over linear segments.
    
    Parameters
    ----------
    qpath : (nq,N) ndarray
        The path nodes of `nq` coordinates at `N` nodes.
    gvib : (N,nq,nq) ndarray
        The effective vibrational metric tensor at each node.

    Returns
    -------
    ds : (N-1,) ndarray
        The arc length of each path segment 

    """
    
    nq,N = qpath.shape 
    
    ds = np.zeros((N-1,)) 
    
    for i in range(N-1):
        # ds[i] 
        # Segment between node i and i + 1
        dq = qpath[:,i+1] - qpath[:,i]          # The path displacement
        gbar = 0.5 * (gvib[i+1] + gvib[i])      # The mean metric tensor
        ds2 = dq @ gbar @ dq                    # The square path length
        ds[i] = np.sqrt(ds2) 
    
    return ds

def _calc_path_length_grad(qpath,gv,Dgv,ds):
    
    # qpath ... (nq,N)      Path 
    # 
    # gv ....... (N,nq,nq)  Metric tensor
    # Dgv ... (nq,N,nq,nq)  Derivatives of metric tensor 
    # ds  ...    (N-1)      The path length segments
    
    # Returns
    # 
    # Dds2_b ... (nq,N-1)   Derivative of ds**2 of segment i w.r.t node before
    # Dds2_a ... (nq,N-1)   " " " node after 
    # Dds_b ... (nq,N-1)   Derivative of ds of segment i w.r.t node before
    # Dds_a ... (nq,N-1)   " " " node after 
    #
    #
    # Calculate the derivative of the ds**2 value
    # with respect to the node before and the node after.
    #
    nq,N = qpath.shape 
    #
    #
    Dds2_b = np.zeros((nq,N-1)) # The of segment i w.r.t the node before
    Dds2_a = np.zeros((nq,N-1)) # The of semgent i wr.t. the node after 
    
    for i in range(N-1):
        #
        # Derivatives of segment i
        #
        # There are two contributions:
        # 1) The derivative of segment displacement
        # 2) the derivative of the mean metric tensor 
        #
        dq = qpath[:,i+1] - qpath[:,i] # The segment path displacement
        gbar = 0.5 * (gv[i] + gv[i+1]) # The mean metric 
        
        Dds2_b[:,i] += -2 * gbar @ dq  # 1) before 
        Dds2_a[:,i] += +2 * gbar @ dq  # 1) after
        
        Dds2_b[:,i] += dq @ (0.5 * Dgv[:,i]) @ dq   # 2) before 
        Dds2_a[:,i] += dq @ (0.5 * Dgv[:,i+1]) @ dq # 2) after
        
    Dds_b = Dds2_b / (2*ds)
    Dds_a = Dds2_a / (2*ds)
    
    return Dds_b, Dds_a 

def _calc_path_length_hes(qpath, gv, Dgv, D2gv, ds, Dds_b, Dds_a):
    
    # qpath ... (nq,N)      Path 
    # 
    # gv ....        (N,nq,nq)  Metric tensor
    # Dgv ...     (nq,N,nq,nq)  Derivatives of metric tensor
    # D2gv ..  (nq,nq,N,nq,nq)  Hessian of metric tensor 
    #
    # ds  ...    (N-1)      The path length segments
    #
    # Dds_b ... (nq,N-1)   Derivative of ds of segment i w.r.t node before
    # Dds_a ... (nq,N-1)   " " " node after 
    
    # Returns
    # 
    # D2ds_bb ...  (nq, nq, N-1)
    # D2ds_ba ...  (nq, nq, N-1)
    # D2ds_aa ...  (nq, nq, N-1)
    #
    #
    # Calculate the derivative of the ds**2 value
    # with respect to the node before and the node after.
    #
    nq,N = qpath.shape 
    #
    #
    D2ds2_bb = np.zeros((nq,nq,N-1)) # The before/before block of the hessian
    D2ds2_ba = np.zeros((nq,nq,N-1)) # The before/after block of the hessian
    D2ds2_aa = np.zeros((nq,nq,N-1)) # The after/after block of the hessian
    
    for i in range(N-1):
        #
        # Second derivatives of segment i
        #
        dq = qpath[:,i+1] - qpath[:,i] 
        gbar = 0.5 * (gv[i] + gv[i+1]) 
        
        # The before/before block 
        #
        D2ds2_bb[:,:,i] += 2 * gbar 
        D2ds2_bb[:,:,i] += dq @ (0.5 * D2gv[:,:,i]) @ dq # contract tensor indices 
        cross = -Dgv[:,i] @ dq 
        D2ds2_bb[:,:,i] += cross + cross.T 
        
        # The after/after block 
        D2ds2_aa[:,:,i] += 2 * gbar 
        D2ds2_aa[:,:,i] += dq @ (0.5 * D2gv[:,:,i+1]) @ dq 
        cross = +Dgv[:,i+1] @ dq 
        D2ds2_aa[:,:,i] += cross + cross.T 
        
        # The before/after block 
        D2ds2_ba[:,:,i] += -2 * gbar 
        D2ds2_ba[:,:,i] += Dgv[:,i] @ dq 
        D2ds2_ba[:,:,i] += -(Dgv[:,i+1] @ dq).T 
    
    D2ds_bb = D2ds2_bb / (2*ds) - np.einsum('in,jn->ijn', Dds_b, Dds_b) / ds
    D2ds_aa = D2ds2_aa / (2*ds) - np.einsum('in,jn->ijn', Dds_a, Dds_a) / ds
    D2ds_ba = D2ds2_ba / (2*ds) - np.einsum('in,jn->ijn', Dds_b, Dds_a) / ds
        
    return D2ds_bb, D2ds_ba, D2ds_aa

def opt_path_bfgs(qpath, V, cs, masses, band_action = 1000.0,
                  kinetic_energy = 0.0,
                  Hinit = None, Vmin = 0.0,
                  alpha = 1.0, tol = 1e-6, maxiter = 100,
                  disp = True, streak = None):
    """
    
    Optimize a least-action path using BFGS Hessian updates. 

    Parameters
    ----------
    qpath : (nq,N) ndarray
        The initial coordinate path of `nq` coordinates along `N` nodes.
    V : DFun
        The potential energy surface supporting `deriv >= 1`.
    cs : CoordSys
        The coordinate system
    masses : array_like
        The masses.
    band_action : float, optional
        The band action force constant. The default is 1000.0.
    kinetic_energy : float, optional
        An energy offset added to the potential. The default is 0.
    Vmin : float, optional
        The minimum potential energy. The default is 0. 
    alpha : float, optional
        The update step scaling coefficint. The default is 1.0.
    tol : float, optional
        The gradient norm tolerance. The default is 1e-6.
    maxiter : int, optional
        The maximum number of steps. The default is 100.
    disp : bool, optional
        Print detailed output. The default is True.
        
    Returns
    -------
    qopt : (nq,N) ndarray
        The optimized path 
    I : float
        The optimized path action 
    s : float
        The optimized path length
    B : float
        The optimized band action

    """

    conv = False     

    if disp:
        print("Step   Action (I)   Band action (B)  Arc Length (s)   |grad(I+B)|         ")
        print("--------------------------------------------------------------------------")
        
        
    qcurr = np.array(qpath).copy() # the initial path 
    
    nq,N = qcurr.shape # The number of coordinates and nodes 
    
    #
    # The first and last node are fixed and will not be optimized
    #
    nH = nq * (N-2) # The number of parameters and the size of the Hessian 
    
    if Hinit is None:
        iHnew = np.eye(nH) # Initial Inverse Hessian 
    else: 
        iHnew = np.linalg.inv(Hinit)
        
    iHold = 0 
    gold = 0 
    dx = 0 
    
    Zero = np.zeros((nq,1))
    
    restart_cnt = 0 
    
    streak_cnt = 0 
    
    for i in range(maxiter):
        #
        #
        # BFGS Hessian update
        #
        # 1) Calculate the current action and derivatives
        s,I,B,Ds,DI,DB = calcPathAction(qcurr, V, cs, masses, 
                                        band_action = band_action,
                                        kinetic_energy=kinetic_energy,
                                        Vmin = Vmin,
                                        deriv = 1)
        
        # Calculate the net gradient and discard the first and last nodes
        g = (DI + DB)
        g = (g[:,1:-1]).reshape((-1,)) # Reshape to 1D 
        
        Ds = (Ds[:,1:-1]).reshape((-1,))
        
        # 2) Check convergence
        gnorm = np.linalg.norm(g)
    
        
        if disp:
            print(f" {i+1:2d}    {I:.4e}     {B:.4e}      {s:.4e}      {gnorm:.4e}  ", end = "")
        
        if gnorm <= tol:
            conv = True 
            break 
        
        # 3) Update Hessian using previous step
        if i > 0:
            dg = g - gold # The gradient difference
            # dx already equals the coordinate difference 
            
            # if i == 1:
            #    gamma0 = np.dot(dg,dx) / np.dot(dg,dg)
            #    iHold *= gamma0 
                
            # Update the inverse Hessian 
            iHdg = iHold @ dg 
            dxdg = np.dot(dx, dg) # dx . dg 
            # Calculate the update
            diH = np.outer(dx,dx) * (dxdg + np.dot(dg, iHdg))/(dxdg**2)
            diH -= (np.outer(iHdg,dx) + np.outer(dx,iHdg)) / dxdg 
            
            iHnew = iHold + diH 
        
        gnorm_old = np.linalg.norm(gold)
        
        # if restart_cnt > 5 and gnorm > 1.5 * gnorm_old:
        #     print("")
        #     print("|g| increased by >50%. Restarting.", end = "")
        #     restart_cnt = 0 
        #     iHnew = np.eye(nH) * gamma0 
        #     iHold = 0 
            
            
        # Streak check for increasing alpha
        # if convergence looks good. 
        if streak is not None:
            if gnorm < gnorm_old:
                streak_cnt += 1
            else:
                streak_cnt = 0 
            
            if streak_cnt >= streak:
                streak_cnt = 0 
                if alpha <= 0.25:
                    alpha *= 2 
                else:
                    alpha = 1 - (1-alpha)/2 
                
                print("")
                print(f"Streak reached. Increasing alpha to {alpha:.2f}", end = "")
        
            
            
        #
        # 4) Calculate displacement
        #    using the updated Hessian 
        dx = -alpha * iHnew @ g 
        
        # 4a) Check the first-order arc-length change
        delta_s = sum(Ds * dx) 
        if abs(delta_s / s) > 0.1:
            # Limit change to 10% at most.
            dx *= 0.1 / abs(delta_s / s)
            print("")
            print("Limiting arc length change to 10%.", end = "")
        
        # 5) Update the geometry
        dq = np.concatenate([Zero, dx.reshape((nq,N-2)), Zero], axis = 1)
        qcurr += dq
        
        iHold = iHnew 
        gold = g 
        
        if disp:
            print("")
            
        restart_cnt += 1 
    #
    ##########################

    if disp:
        print("")
        if conv:
            print(f"Convergence reached, |g| = {gnorm:.3e}")
        else:
            print(f"Maximum iterations reached, |g| = {gnorm:.3e}")


    # Return the final path, action, arc length, and band energy
    
    
    return qcurr, I, s, B


def calcWKBtunneling(qpath, V, cs, masses, disp = True,
                     hbar = None):
    """
    Perform a simple WKB tunneling analysis on a 1D path.

    Parameters
    ----------
    qpath : (nq,N)
        The path between two equivalent minima.
    V : DFun or function
        The potential energy
    cs : CoordSys
        The coordinate system.
    masses : array_like
        The masses.
    disp : bool, optional
        Display/print flag. The default is True.
    hbar : float, optional
        The value of hbar. If None, default NITROGEN units are used.

    Returns
    -------
    omega : float
        The estimated harmonic energy 
    phi : float
        The imaginary action in units of :math:`\\hbar`. 
    dE : float
        The tunneling energy splitting

    """
    
    nq,N = qpath.shape 
    if hbar is None:
        hbar = nitrogen.constants.hbar 
        
    # Compute action 
    
    # Calculate the potential energy and its derivatives 
    V = V.f(qpath, deriv = 0)[0,0] 
    V -= min(V) # Subtract minimum energy 
    
    # Calculate the metric tensor
    g = cs.Q2g(qpath, masses = masses, deriv = 0)[0] # (npacked, N) 
    gfull = nitrogen.linalg.packed.symfull(g) # (nq+3,nq+3,N)
    gfull = np.moveaxis(gfull, [0,1],[-2,-1])  # (N, nq+3, nq+3)
    Gfull = np.linalg.inv(gfull)
    gvib = np.linalg.inv(Gfull[..., :nq, :nq]) 
    
    # Calculate the path length of each segment
    ds = _calc_path_length_trap(qpath, gvib)
    si = np.concatenate([[0.], np.cumsum(ds)])
    
    if disp:
        plt.figure() 
        plt.plot(si, V, 'm.-')
        plt.xlabel('Mass-weighted path length')
        plt.ylabel('Energy')
        
    # Calculate the approximate zero-point energy 
    # First, calculate the quadratic force constant
    # w.r.t. the mass-weighted path length 
    #
    k = 2 * (V[1] - V[0]) / (ds[0])**2 
    
    # The harmonic frequency is the sqrt of this.
    omega = hbar * np.sqrt(k) 
    zpe = omega / 2 
    if disp:
        print(f"The harmonic frequency is {omega:.4f}")
        plt.plot([si[0], si[-1]], [zpe, zpe], 'k--')
        
    # Calculate the imaginary action 
    pi = np.sqrt(2 * (V - zpe) + 0*1j) 
    I = 0.0 
    for i in range(N-1):
        I += ds[i] * (pi[i] + pi[i+1]) / 2.0 
    
    phi = np.real(I/hbar)
    
    # The WKB tunneling estimate
    dE = np.exp(-phi)/np.pi * omega 
    
    if disp:
        print(f"The imaginary action is I = {phi:.4f} x hbar")
        print(f"The WKB tunneling splitting is dE = {dE:.4e}")
    
    return omega, phi, dE 
    
    



        
    
    