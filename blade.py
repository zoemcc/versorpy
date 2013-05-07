import numpy as np
import scipy.linalg as la
from contracts import contract

"""
    A blade class and operations on blades in n-dimensional spaces.
    This library is adapted from Daniel Fontijne's PhD thesis,
    "Efficient Implementation of Geometric Algebra"
    staff.science.uva.nl/~fontijne/phd.html
    referenced here as DFPhD
"""


class Blade(object):
    @contract
    def __init__(self, blade=None, orthonormal=False, s=None, gpu=False, 
                 tol=1e-8, copy=True):
        """
            Creates a new blade.

            :param blade: A matrix spanning the columns of the blade.
            :type blade: array[NxK]|array[N]|number|None

            :param orthonormal: Is the input blade already orthonormal?
            :type orthonormal: bool|None

            :param s: A number to scale the result by.
            :type s: number|None

            :param gpu: Should the blade be stored on the gpu and computations
                        involving the blade be performed on the gpu?
            :type gpu: bool

            :param tol: Any number below tol is treated as 0 for rank
                        calculations
            :type tol: float

            :param copy: Should we copy the input array instead of storing a pointer?
            :type copy: bool
        """
        self.gpu = gpu # gpu not supported yet -- stay tuned!

        if blade is None:
            self.s = 0
        else:
            try:
                shape = blade.shape
                if copy:
                    blade = np.copy(blade)
                if len(shape) == 1: #only one index, make column vector
                    blade = np.array([blade]).T
                    shape = blade.shape
                elif shape[1] == 0:
                    self.s = s if s is not None else 1
            except AttributeError: #try scalar?
                self.s = s * blade if s is not None else blade
                blade = np.array([[]])
                shape = blade.shape
            except:
                raise
            n, k = shape
            self.n, self.k = n, k
            if k == 0: # scalar
                self.blade = blade
            elif n < k: # too many columns, can't be a k-dim subspace of R^n
                self.s = 0
            elif orthonormal: # no qr/rescaling required -- known to be ortho
                self.s = s if s is not None else 1
                self.blade = blade
            elif np.allclose(np.dot(blade.T, blade), np.eye(k)):
                # blade has been discovered to be orthonormal
                self.s = s if s is not None else 1
                self.blade = blade
            elif k == 1: # just a vector
                mag = la.norm(blade)
                if abs(mag) >= tol:
                    self.s = s * mag if s is not None else mag
                    self.blade = blade / mag
                else:
                    self.s = 0
            else: # full blade case
                # QR factorization
                Q, R, P = la.qr(blade, mode='economic', pivoting=True)
                print Q, R, P
                print np.dot(Q.T, Q)
                print 'k: ', k
                print R[k - 1, k - 1]
                if abs(R[k - 1, k - 1]) >= tol:
                   self.s = np.prod(np.diag(R))
                   if s is not None: self.s *= s
                   self.blade = Q
                   print 'self.s: ', self.s
                   print 'self.blade: ', self.blade
                else:
                   self.s = 0
        if np.allclose(self.s, 0): # reset to zero blade
            self.n = 1 # necessary for consistency with blade.shape == (1, 0)
            self.k = 0
            self.s = 0
            self.blade = np.array([[]])

def outer(blade1, blade2, tol=1e-10):
    """ Calculates the blade outer product of blade1 ^ blade2 """
    # TODO: implement operator overloading for ^
    print 'blade1: ', blade1.blade
    print 'blade2: ', blade2.blade
    k, l = blade1.k, blade2.k
    if k == 0 and l == 0: # scalar/scalar
        return Blade(1, s = blade1.s * blade2.s)
    elif k == 0: # scalar/blade
        return Blade(blade2.blade, orthonormal=True, s=blade1.s * blade2.s)
    elif l == 0: # blade/scalar
        return Blade(blade1.blade, orthonormal=True, s=blade1.s * blade2.s)
    elif k + l > blade1.blade.shape[0]: # too many factors
        return Blade()
    else: # full calculation
        O = np.concatenate((blade1.blade, blade2.blade), axis=1)
        U, s, Vh = la.svd(O)
        if abs(s[k + l - 1]) < tol:
            return Blade()
        else:
            matrixSpan = U[:, :k + l]
            scale = blade1.s * blade2.s * la.det(np.dot(O.T, matrixSpan))
            print 'Span: ', matrixSpan
            print 'O: ', O
            print 'scale: ', scale
            return Blade(blade=matrixSpan, orthonormal=True, s=scale)

def inverse(blade):
    """ Calculates the inverse of k-blade blade  """
#TODO: make inplace work here as an argument and doesn't mess things up
#TODO: make this work for arbitrary metric
    if blade.s == 0:
        raise AttributeError, ('Not invertible, s=0', blade)
    else:
        if blade.k == 0:
            return Blade(1, s=1.0 / blade.s)
        else:
            revBlade    = reverse(blade, inplace=False)
            denominator = inner(revBlade, blade)
            revBlade.s /= (denominator.s + np.finfo(np.double).eps)
            return revBlade

def inverseScaling(blade, inplace=False):
    """ Calculates the inverse of k-blade blade and returns just scaling """
#TODO: currently broken WARNING
    revBlade    = reverse(blade)
    denominator = inner(revBlade, blade)
    revBlade.s /= (denominator.s + np.finfo(np.double).eps)
    return revBlade.s


def copy(blade, s=None):
    """ s allows the scale to be set independently of the input blade """
    scale = s if s is not None else blade.s
    return Blade(blade=blade.blade, orthonormal=True, s=scale, 
                 gpu=blade.gpu, copy=True)


def reverse(blade, inplace=True):
    """ 
    Calculates the reverse of k-blade blade.
    If inplace is false, return just s, else return the modified blade.
    """
    if not inplace:
        blade = copy(blade)
    odd = ((blade.k * (blade.k - 1)) / 2) % 2
    blade.s *= (-1)**odd
    return blade

def reverseScaling(blade, inplace=False):
    """
    Calculates the grade involution of k-blade blade and returns just scaling.
    """
    odd = ((blade.k * (blade.k - 1)) / 2) % 2
    s = blade.s
    s *= (-1)**odd
    if inplace:
        blade.s *= (-1)**odd
    return s

def involution(blade, inplace=True):
    """
    Calculates the grade involution of k-blade blade  
    If inplace is false, return just s, else return the modified blade.
    """
    if not inplace:
        blade = copy(blade)
    odd = blade.k % 2
    blade.s *= (-1)**odd
    return blade

def involutionScaling(blade, inplace=False):
    """
    Calculates the grade involution of k-blade blade and returns just scaling.
    """
    odd = blade.k % 2
    s = blade.s
    s *= (-1)**odd
    if inplace:
        blade.s *= (-1)**odd
    return s


def inner(blade1, blade2):
    """ 
    Calculates the blade inner product of blade1 \circdot blade2.
    Not yet implemented for non-euclidean inner products.
    """
    k1, k2 = blade1.k, blade2.k
    if k1 != k2:
        return Blade()
    else:
        if k1 == 0:
            # needs to multiply by the blade scales.  DFPhD assumes unit blades.
            scale = blade1.s * blade2.s * \
                    (-1)**((k1 * (k1 - 1)) / 2)
        else:
            # needs to multiply by the blade scales.  DFPhD assumes unit blades.
            scale = blade1.s * blade2.s * \
                    (-1)**((k1 * (k1 - 1)) / 2) * \
                    la.det(np.dot(blade1.blade.T, blade2.blade))
        C = Blade(1, s=scale)
        return C

def pseudoScalar(n):
    """ Return the pseudoscalar of the space R^n """
    return Blade(blade=np.eye(n, n))

def pseudoScalarReverse(n):
    """ Return the reverse of the pseudoscalar of the space R^n """
    odd = ((n * (n - 1)) / 2) % 2
    return Blade(blade=np.eye(n, n), s=(-1)**odd)

def dual(blade, n=None):
    """ Calculates the dual of blade with respect to the euclidean metric """
    if blade.s == 0:
        raise AttributeError, ('Not dualizable, s=0', blade)
    else:
        if n is None:
            n = blade.n
        k = blade.k
        if k == 0: #return an n-blade
            retBlade = pseudoScalarReverse(n)
            retBlade.s *= blade.s
        elif k == n: # return a 0-blade (scalar)
            scale = blade.s * la.det(blade.blade)
            retBlade = Blade(1, s=scale)
        else:
            Q, R = la.qr(blade.blade, mode='full')
            D = Q[:, k:] # take the orthogonal complement to A
            # typo in DFPhD -- should be [A, D], not [D, A] as listed
            O = np.concatenate((blade.blade, D), axis=1) 
            odd = ((k * (k - 1) + n * (n - 1)) / 2) % 2
            # needs to multiply by the blade scale.  DFPhD assumes unit blade.
            scale = blade.s * (-1)**odd * la.det(O)
            retBlade = Blade(D, s=scale, orthonormal=True)
        return retBlade

def undual(blade, n=None):
    """ Calculates the undual of blade with respect to the euclidean metric """
    retBlade = dual(blade, n=n)
    n = n if n is not None else blade.n
    odd = (n * (n - 1) / 2) % 2
    retBlade.s *= (-1)**odd
    return retBlade


def dualNonEuc(blade, n=None, metric=('conformal'), metMatrix=None):
    """ Calculates the dual of blade with respect to the metric matrix """
    # WARNING: not yet finished. I'm putting off conformal transformations 
    # in order to focus initially on rotations
    if blade.s == 0:
        raise AttributeError, ('Not dualizable, s=0', blade)
    else:
        # non euclidean stuff
        if metric[0] == 'conformal':
            M = np.eye(n + 2)
            M[0, 0] = 0
            M[-1, -1] = 0
            M[-1, 0] = -1
            M[0, -1] = -1
        else:
            M = metMatrix
        if n is None:
            n = blade.n
        k = blade.k
        if k == 0: #return an n-blade
            retBlade = pseudoScalarReverse(n)
            retBlade.s *= blade.s
        elif k == n: # return a 0-blade (scalar)
            scale = blade.s * la.det(blade.blade)
            retBlade = Blade(1, s=scale)
        else:
            # non euclidean stuff
            transformedBlade = np.dot(M, blade.blade)
            Q, R = la.qr(transformedBlade, mode='full')
            D = Q[:, k:] # take the orthogonal complement to A
            # typo in DFPhD -- should be [MA, D], not [D, MA] as listed
            O = np.concatenate((transformedBlade, D), axis=1) 
            odd = ((k * (k - 1) + n * (n - 1)) / 2) % 2
            scale = blade.s * (-1)**odd * la.det(O)
            retBlade = Blade(D, s=scale, orthonormal=True)
        return retBlade

def leftContract(blade1, blade2):
    """
    Calculate the left contraction.
    Projects blade1 onto blade2 and then 
    then takes the orthogonal complement of the 
    projected blade1 within blade2.
    """
    if blade1.k == blade2.k:
        return inner(blade1, blade2)
    elif blade1.k > blade2.k:
        return Blade()
    else:
        n = blade2.n
        return undual(outer(blade1, dual(blade2, n=n)), n=n)

def equality(blade1, blade2):
    """
    Calculates whether or not blade1 is the same blade as blade2,
    including scale and orientation.
    """
    if blade1.k != blade2.k:
        return False
    elif blade1.k == 0:
        print 'blade1.k = 0'
        return np.allclose(blade1.s, blade2.s)
    elif join(blade1, blade2).k == blade1.k:
        print 'join = k', blade1.k
        return np.allclose(inner(inverse(blade1), blade2).s, 1)
    else:
        return False

def subSpaceEquality(blade1, blade2):
    """
    Calculates whether or not blade1 represents the same subspace as blade2,
    not including scale and orientation.
    """
    if blade1.k != blade2.k:
        return False
    elif blade1.k == 0 or blade1.k == blade1.n:
        return True
    elif join(blade1, blade2).k == blade1.k:
        return True
    else:
        return False


def join(blade1, blade2, tol=1e-8):
    """ 
    Compute the join of blade1 and blade2 upto tolerance tol.
    The scale is set to be 1 regardless of input scale since 
    the join has no absolute magnitude.
    """
    if blade1.k == 0 and blade2.k == 0:
        return Blade(1, s=1)
    if blade1.k == 0:
        ret = copy(blade2, s=1)
        return ret
    elif blade2.k == 0:
        ret = copy(blade1, s=1)
        return ret
    else:
        combinedMat = np.concatenate((blade1.blade, blade2.blade), axis=1)
        U, s, Vt = la.svd(combinedMat)
        for (i, si) in enumerate(s):
            if si <= tol:
                j = i
                break
        else:
            j = s.shape[0]
        joinMat = U[:, :j]
        print "joinMat: ", joinMat
        print 's: ', s
        print 'j: ', j
        return Blade(joinMat, s=1, orthonormal=True, copy=False)

def meet(blade1, blade2, tol=1e-8):
    """ 
    Compute the meet of blade1 and blade2 upto tolerance tol.
    The scale is set to be 1 regardless of input scale since 
    the meet has no absolute magnitude.
    """
    ret = leftContract(leftContract(blade2, inverse(join(blade1, blade2, \
                       tol=tol))), blade2)
    ret.s = 1
    return ret

def addition(blade1, blade2):
    """
    Can only be performed if blade1.k == blade2.k
    and meet(blade1, blade2).k >= k - 1.
    Not yet implemented.
    """
    # TODO: all of this
    pass

def applyLinearTransform(transform, blade):
    """
    Applies the linear transformation transform to blade 
    """
    if blade.k == 0:
        return Blade(1, s=blade.s * transform)
    else:
        return Blade(np.dot(transform, blade.blade), s=blade.s, copy=False)






    




