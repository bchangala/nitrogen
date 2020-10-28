"""
zmat.py

Z-matrix (ZMAT) coordinate system

"""
from .coordsys import CoordSys 
import nitrogen.autodiff.forward as adf
import numpy as np

__all__ = ['ZMAT'] # only provide ZMAT class for import *

class ZMAT(CoordSys):
    """
    Z-matrix coordinate system.
    
    Angles are in *degrees*, not radians. (This is a
    different convention than NITROGEN 1.x.)
    
    Attributes
    ----------
    zmat : str
        A Z-matrix string that reproduces this ZMAT.
    angles : {'deg','rad'}
        The units (degrees or radians) used for angular coordinates.
        
    """
    
    def __init__(self, zmatrix, angles = 'deg'):
        """
        Create a new ZMAT CoordSys.
        
        Parameters
        ----------
        zmatrix : str
            Z-matrix definition string.
        angles : {'deg','rad'}
            Angles are interpreted as degrees ('deg') or radians ('rad').
            The default is 'deg'.

        """
        
        # Parse the zmatrix string
        atomLabels, ref, val, coordID, coordLabels, zmat = parseZMAT(zmatrix)

        
        # Determine the dummy atoms
        notDummy = []
        natoms = 0
        for label in atomLabels:
            if label[0].upper() == 'X':
                notDummy.append(False)
            else:
                notDummy.append(True)
                natoms += 1
        if natoms < 1:
            raise ValueError("ZMAT must have at least one non-dummy atom")
        nX = 3 * natoms 
        
        
        nQ = len(coordLabels)
        if nQ < 1 :
            raise ValueError("ZMAT must have at least one coordinate")
        
        #######################################
        # Initialize CoordSys
        super().__init__(self._zmat_q2x, nQ, nX, name = 'ZMAT',
                         Qstr = coordLabels, isatomic = True)
        # new attributes
        self._atomLabels = atomLabels    # Label for each atom of Z-matrix
        self._notDummy = notDummy        # Dummy/not-dummy 
        self._ref = ref                  # Reference atoms for coordinates (1-indexed)
        self._val = val                  # Coordinate values or coord coeffs.
        self._coordID = coordID          # CS coordinate ID for each ZMAT coordinate
        self.zmat = zmat                 # New uniformized z-matrix string
        self.angles = angles             # Angle unit ('deg' or 'rad')
        
    def _zmat_q2x(self, Q, deriv = 0, out = None, var = None):
        """
        ZMAT X(Q) function.

        Parameters
        ----------
        Q : ndarray
            Input coordinate array (`self.nQ`, ...)
        deriv : int, optional
            Maximum derivative order. The default is 0.
        out : ndarray, optional
            Output location (nd, `self.nX`, ...) . The default is None.
        var : list of int, optional
            Requested ordered derivative variables. If None, all are used.

        Returns
        -------
        out : ndarray
            The X coordinates.

        """
        
        natoms = self.natoms        # Number of non-dummy atoms
        base_shape =  Q.shape[1:]
        
        if var is None:
            var = [i for i in range(self.nQ)]
        nvar = len(var)             # The number of derivative variables
        
        nd = adf.nderiv(deriv, nvar) # The number of derivatives
        
        # Set up output location
        if out is None:
            out = np.ndarray( (nd, 3*natoms) + base_shape, dtype = Q.dtype)
        out.fill(0) # Initialize out to 0
    
        
        # Create adf symbols/constants for each CS coordinate
        q = [] 
        for i in range(self.nQ):
            if i in var: # Derivatives requested for this variable
                q.append(adf.sym(Q[i], var.index(i), deriv, nvar))
            else: # Derivatives not requested, treat as constant
                q.append(adf.const(Q[i], deriv, nvar))

        # Calculate ZMAT atomic Cartesian positions, A
        #
        nA = len(self._atomLabels) # The number of ZMAT entries, including dummy
        A = np.ndarray((nA,3), dtype = adf.adarray) # ZMAT atom positions
        Ci = np.ndarray((3,), dtype = adf.adarray)  # ZMAT coordinate for row i
        
        # For each atom (dummy + non-dummy) in the ZMAT
        for i in range(nA):
            
            # First, calculate the current row's ZMAT coordinates
            # C = [r, theta, tau]
            for j in range(3):
                if j - i >= 0:
                    # ZMAT coordinate not defined; will not be referenced
                    continue
                if self._coordID[i][j] is None:
                    # C[i,j] is a constant literal value
                    Ci[j] = adf.const(np.full(base_shape, self._val[i][j]), deriv, nvar)
                else:
                    # A value*coordinate entry
                    # Use coordID[i,j] to lookup the correct CS coordinate
                    # adarray in q
                    Ci[j] = self._val[i][j] * q[self._coordID[i][j]]
                
                # Convert angular coordinates to correct unit
                if j == 1 or j == 2:
                    if self.angles == 'rad':
                        pass # already correct
                    elif self.angles == 'deg':
                        Ci[j] = Ci[j] * (np.pi/180.0) # convert deg to rad
                    else:
                        raise ValueError("Invalid angle units")
            
            # Now calculate the position of atom i, A[i]
            #
            # There are several special cases:
            # Case 1: the first atom is at the origin
            if i == 0:
                for j in range(3):
                    A[0,j] = adf.const(np.zeros(base_shape), deriv, nvar)
            
            # Case 2: the second atom is on the +z axis
            elif i == 1:
                # x and y are 0
                A[1,0] = adf.const(np.zeros(base_shape), deriv, nvar) # x
                A[1,1] = adf.const(np.zeros(base_shape), deriv, nvar) # y
                #
                # The z-value is given by r = Ci[0]
                A[1,2] = Ci[0].copy()
            
            # Case 3: the third atom is in the z/+x plane
            elif i == 2:
                # There are two possible sub-cases:
                # a) References are 1, then 2
                # b) References are 2, then 1
                #
                # In both cases, y = 0 and
                # x = r * sin(theta * pi/180 )
                A[2,1] = adf.const(np.zeros(base_shape), deriv, nvar) # y
                A[2,0] = Ci[0] * adf.sin(Ci[1])                       # x
                # 
                # the z coordinate depends on the sub-case
                if self._ref[2] == (1,2,None): # Case 3a)
                    #     1---2   +---> z
                    #    /        |
                    #   @       x v
                    #             
                    # z = r * cos(theta * pi/180)
                    A[2,2] = Ci[0] * adf.cos(Ci[1])

                elif self._ref[2] == (2,1,None): # Case 3b)
                    #  1---2       +---> z
                    #       \      |
                    #        @   x v
                    #
                    # z = z2 - r*cos(theta*pi/180)
                    A[2,2] = A[1,2] - Ci[0] * adf.cos(Ci[1])
                    
                else:
                    # Should not reach here, the references are invalid
                    raise ValueError("Invalid ZMAT reference evaluation")
            
            # Case 4: the general case for the fourth and later atoms
            else:
                #
                #
                # This atom (A[i]) is connected sequentially 
                # to its references: i -> ref[0] -> ref[1] -> ref[2]
                A1 = A[self._ref[i][0] - 1] #`ref` contains 1-indexed values
                A2 = A[self._ref[i][1] - 1]
                A3 = A[self._ref[i][2] - 1]
                
                r21 = A2 - A1 
                r32 = A3 - A2 
                rC = np.cross(r21, r32)
                
                # Construct the local coordinate system axes
                U = r21 / adf.sqrt(r21[0]*r21[0] + r21[1]*r21[1] + r21[2]*r21[2])
                V = rC / adf.sqrt(rC[0]*rC[0] + rC[1]*rC[1] + rC[2]*rC[2])
                W = np.cross(U, V)
                
                r = Ci[0] 
                th = Ci[1]
                phi = Ci[2]
                
                for j in range(3):
                    A[i,j] = A1[j] \
                        + r*adf.cos(th) * U[j] \
                        - r*adf.sin(th)*adf.sin(phi) * V[j] \
                        - r*adf.sin(th)*adf.cos(phi) * W[j]
                
                
        ##################################################
        #
        # All atom positions A are now calculated
        # 
        # Place non-dummy derivative arrays in output
        k = 0
        for i in range(nA):
            if self._notDummy[i]: # not a dummy
                # output atom k <-- ZMAT atom i
                for j in range(3):
                    np.copyto(out[:,3*k+j], A[i,j].d )
                k += 1
        # done!
        
        return out
    
    def __repr__(self):
        return f'ZMAT({self.zmat!r})'
    

    def diagram(self): 
        # using U+250X box and U+219X arrows
        diag = ""
        
        diag += "     │↓              ↑│        \n"
        diag += "     │Q              X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ │ ┌────────────┐ │ ║      \n"
        diag += "   ║ ╰─┤  Z-matrix  ├─╯ ║      \n"
        diag += "   ║   └────────────┘   ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
                    
        
def parseZMAT(ZMAT):
    """
    Parse a ZMAT text string.

    Parameters
    ----------
    ZMATstr : str
        The ZMAT specification.

    Returns
    -------
    None.

    """
    
    atomLabels = []
    ref = []
    val = []
    coordID = []
    coordLabels = []
    
    # Split into stripped lines, ignoring anything past '#'
    lines = [i.split('#')[0].strip() for i in ZMAT.splitlines()]
    # Keep only non-empty lines
    lines = list(filter(len,lines))
    
    # Line 1
    if len(lines) > 0:
        items = lines[0].split() # We expect only 1 item, the atom label
        if len(items) < 1:
            raise ValueError("First line of ZMAT must have 1 item")
        atomLabels.append(items[0])
        # The first line has no references, values, or coordinates
        ref.append((None,None,None))
        val.append((None,None,None))
        coordID.append((None,None,None))
    else: 
        raise ValueError("ZMAT must have at least one entry")
        
    # Line 2
    if len(lines) > 1:
        items = lines[1].split() # We expect 3 items
        if len(items) < 3:
            raise ValueError("Second line of ZMAT must have 3 items")
        atomLabels.append(items[0])
        
        ref.append((int(items[1]), None, None)) # There is one reference atom
        v,c = parseValueCoord(items[2])         # Parse the value/coordinate
        val.append((v, None, None))
        if c is not None:
            coordLabels.append(c)
            c = coordLabels.index(c)        
        coordID.append((c, None, None)) # c is still None or the correct index

    
    # Line 3
    if len(lines) > 2:
        items = lines[2].split() # We expect 5 items
        if len(items) < 5:
            raise ValueError("Third line of ZMAT must have 5 items")
        atomLabels.append(items[0])
        
        ref.append((int(items[1]), int(items[3]), None))
        
        v1,c1 = parseValueCoord(items[2])
        v2,c2 = parseValueCoord(items[4])
        
        val.append((v1,v2,None))
        
        if c1 is not None:
            if c1 not in coordLabels:
                coordLabels.append(c1)
            c1 = coordLabels.index(c1)
        if c2 is not None:
            if c2 not in coordLabels:
                coordLabels.append(c2)
            c2 = coordLabels.index(c2)
            
        coordID.append((c1, c2, None))
        
    # Lines 4 and above
    for i in range(3,len(lines)):
        items = lines[i].split() # We expect 7 items
        if len(items) < 7:
            raise ValueError("Line {:d} of ZMAT must have 7 items".format(i+1))
        atomLabels.append(items[0])
        
        ref.append((int(items[1]), int(items[3]), int(items[5])))
        
        v1,c1 = parseValueCoord(items[2])
        v2,c2 = parseValueCoord(items[4])
        v3,c3 = parseValueCoord(items[6])
        
        val.append((v1,v2,v3))
        
        if c1 is not None:
            if c1 not in coordLabels:
                coordLabels.append(c1)
            c1 = coordLabels.index(c1)
        if c2 is not None:
            if c2 not in coordLabels:
                coordLabels.append(c2)
            c2 = coordLabels.index(c2)
        if c3 is not None:
            if c3 not in coordLabels:
                coordLabels.append(c3)
            c3 = coordLabels.index(c3)
            
        coordID.append((c1, c2, c3))
    
    # All entries have been parsed. Complete some remaining value checks

    # All references must be to previous atoms
    # and none can be repeated in a given line
    for i in range(len(ref)):
        for j in range(3):
            if ref[i][j] is None: # okay
                continue 
            elif ref[i][j] > i or ref[i][j] <= 0:
                raise ValueError("Invalid ZMAT reference in line {:d} ('{:s}').".format(i+1,lines[i]))
            for k in range(j):
                if ref[i][k] is None:
                    continue
                if ref[i][k] == ref[i][j]: # Two identical references!
                    raise ValueError("Duplicate ZMAT reference in line {:d}.".format(i+1))
        
    
    
    newZMAT = "\n".join(lines)
    
    return (atomLabels, ref, val, coordID, coordLabels,newZMAT)
        
        
def parseValueCoord(vcstr):
    """
    Parse a value*coord ZMAT string.
    
    E.g., "R", "1.0", "2*A", "-2.3*R", "-A"
    
    All whitespace is removed from string before parsing.
    
    If there is no '*' operator, then the string must be
    a valid float literal or a coordinate label. A coordinate
    label must begin with a letter. A leading '+' or '-' may
    be added before a coordinate label (if there is no '*').
    
    If there is a '*' operator, then the string before it must
    be a valid float literal. The string after it must be a valid
    coordinate label.

    Parameters
    ----------
    vcstr : str
        Value-coordinate string

    Returns
    -------
    v : float
        The literal value or coefficient.
    c : str
        The coordinate label. This is None if there is no coordinate.
    """
    
    vcstr = "".join(vcstr.split()) # We expect no whitespace. Remove if any.
    
    if "*" in vcstr:
        # A two factor value * coordinate string
        vstr, cstr = vcstr.split('*')
        if len(vstr) == 0 or len(cstr) == 0:
            raise ValueError("Malformed value-coordinate string")
        elif not cstr[0].isalpha():
            raise ValueError("Invalid coordinate label")
        else:
            return (float(vstr), cstr)
    else:
        # A bare literal or a bare coordinate, possibly with leading '-' or '+'
        if vcstr[0].isalpha():
            return (1.0, vcstr[0:])
        elif vcstr[0] == '-' and vcstr[1].isalpha():
            return (-1.0, vcstr[1:])
        elif vcstr[0] == '+' and vcstr[1].isalpha():
            return (+1.0, vcstr[1:])
        else:
            return (float(vcstr), None)
        
    
    