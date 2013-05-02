import numpy as np
import scipy.linalg as la
from contracts import contract


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
            except AttributeError: #try scalar?
                self.s = s * blade if s is not None else blade
                blade = np.array([[]])
                shape = blade.shape
            except:
                raise
            if len(shape) == 1: #only one index, make column vector
                blade = np.array([blade]).T
                shape = blade.shape
            n, k = shape
            self.n, self.k = n, k
            if k == 0:
                self.blade = blade
            elif orthonormal:
                self.s = s if s is not None else 1
                self.blade = blade
            else:
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
            self.n = 0
            self.k = 0
            self.s = 0
            self.blade = np.array([[]])

def outer(blade1, blade2):
    """ Calculates the blade outer product of blade1 ^ blade2 """
    # TODO: implement operator overloading for ^
    k, l = blade1.k, blade2.k
    if k == 0:
        return Blade(blade2.blade, orthonormal=True, s=blade1.s * blade2.s)
    elif l == 0:
        return Blade(blade1.blade, orthonormal=True, s=blade1.s * blade2.s)
    elif k + l > blade1.blade.shape[0]:
        return Blade()
    else:
        O = np.concatenate((blade1.blade, blade2.blade), axis=1)
        U, s, Vh = la.svd(O)
        # TODO: finish this case




def inverse(blade, inplace=True):
    """ Calculates the inverse of k-blade blade  """
    revBlade    = reverse(blade, inplace=inplace)
#TODO: make sure that inplace doesn't mess things up here
    denominator = inner(revBlade, blade)
    revBlade.s /= (denominator.s + np.finfo(np.double).eps)
    return revBlade

def inverseScaling(blade, inplace=False):
    """ Calculates the inverse of k-blade blade and returns just scaling """
    revBlade    = reverse(blade)
    denominator = inner(revBlade, blade)
    revBlade.s /= (denominator.s + np.finfo(np.double).eps)
    return revBlade.s


def copy(blade):
    return Blade(blade=blade.blade, orthonormal=True, s=blade.s, 
                 gpu=blade.gpu, copy=True)


def reverse(blade, inplace=True):
    """ 
    Calculates the reverse of k-blade blade.
    If inplace is false, return just s, else return the modified blade.
    """
    if not inplace:
        blade = copy(blade)
    even = ((blade.k * (blade.k - 1)) / 2) % 2
    if even:
        blade.s *= -1
    return blade

def reverseScaling(blade, inplace=False):
    """
    Calculates the grade involution of k-blade blade and returns just scaling.
    """
    even = ((blade.k * (blade.k - 1)) / 2) % 2
    s = blade.s
    if even:
        s *= -1
        if inplace:
            blade.s *= -1
    return s

def involution(blade, inplace=True):
    """
    Calculates the grade involution of k-blade blade  
    If inplace is false, return just s, else return the modified blade.
    """
    if not inplace:
        blade = copy(blade)
    even = blade.k % 2
    if even:
        blade.s *= -1
    return blade

def involutionScaling(blade, inplace=False):
    """
    Calculates the grade involution of k-blade blade and returns just scaling.
    """
    even = blade.k % 2
    s = blade.s
    if even:
        s *= -1
        if inplace:
            blade.s *= -1
    return s


def inner(blade1, blade2):
    """ Calculates the blade inner product of blade1 \circdot blade2 """
    k1, k2 = blade1.k, blade2.k
    if k1 != k2:
        return Blade()
    else:
        scale = blade1.s * blade2.s * \
                (-1)**((k1 * (k1 - 1)) / 2) * \
                la.det(np.dot(blade1.blade.T, blade2.blade))
        C = Blade(1, s=scale)
        return C
    




    

                









    
