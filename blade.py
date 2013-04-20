import numpy as np
import scipy.linalg as la


class Blade(object):
    def __init__(self, blade=None, orthonormal=False, s=None, gpu=False, tol=1e-8):
        """
        Inputs:
        blade       : n by F np array that is the list of k vectors of 
                      dimension n to be spanned
        orthonormal : specify whether or not the input B is already orthonormal
        s           : scale factor multiplier (will be multiplied to 
                      the calculated s if input B is not orthnormal
        gpu         : should the blade be stored on the gpu and computations 
                      involving the blade be performed on the gpu?
        """
        self.gpu = gpu # gpu not supported yet -- stay tuned!

        if blade is None:
            self.s = 0
        else:
            try:
                shape = blade.shape
            except AttributeError: #try scalar?
                blade = np.array([[blade]])
                shape = blade.shape
            except:
                raise
            if len(shape) == 1: #only one index, make column vector
                blade = np.array([blade]).T
            n, k = blade.shape
            self.k = k
            if orthonormal:
                self.s = s if s is not None else 1
                self.blade = blade
            else:
                # QR factorization
                Q, R, P = la.qr(blade, mode='economic', pivoting=True)
                print Q, R, P
                print np.dot(Q.T, Q)
                print k
                print R[k - 1, k - 1]
                if abs(R[k - 1, k - 1]) >= tol:
                   self.s = np.prod(np.diag(R))
                   if s is not None: self.s *= s
                   self.blade = np.copy(Q)
                   print 'self.s: ', self.s
                   print 'self.blade: ', self.blade
                else:
                   self.s = 0
        if np.allclose(self.s, 0): # reset to null blade
            self.k = 0
            self.s = 0
            self.blade = np.array([])

def outer(blade1, blade2):
    """ Calculates the blade outer product of blade1 ^ blade2 """
    # TODO: implement operator overloading for ^
    pass

def reverse(blade):
    """ Calculates the reverse of k-blade blade  """
    blade.s *= pow(-1, (blade.k * (blade.k - 1)) / 2)
    return blade

def involution(blade):
    """ Calculates the grade involution of k-blade blade  """
    blade.s *= pow(-1, blade.k)
    return blade

def inner(blade1, blade2):
    """ Calculates the blade inner product of blade1 \circdot blade2 """
    k = blade1.k
    scale = (-1)**((k * (k - 1)) / 2) * \
            la.det(np.dot(blade1.blade.T, blade2.blade))
    C = Blade(blade=1, s=scale)
    return C
    




    

                









    
