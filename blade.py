import numpy as np
import scipy.linalg as la


class Blade(object):
    def __init__(self, B=None, orthonormal=False, s=None, gpu=False, tol=1e8):
        """
        Inputs:
        B           : n by k np array that is the list of k vectors of 
                      dimension n to be spanned
        orthonormal : specify whether or not the input B is already orthonormal
        s           : scale factor multiplier (will be multiplied to 
                      the calculated s if input B is not orthnormal
        gpu         : should the blade be stored on the gpu and computations 
                      involving the blade be performed on the gpu?
        """
        self.gpu = gpu # gpu not supported yet -- stay tuned!

        if B is None:
            self.k = 0
            self.s = s if s is not None else 1
            self.B = np.array([])
        else:
            n, k = B.shape
            if orthonormal:
                self.k = k
                self.s = s if s is not None else 1
                self.B = B
            else:
                # QR factorization
                Q, R, P = la.qr(B, mode='economic', pivoting=True)
                if abs(R[k - 1, k - 1]) >= tol:
                   self.k = k
                   self.s = np.prod(np.diag(R))
                   if s is not None: self.s *= s
                   self.B = Q
                else:
                   self.k = 0
                   self.s = s if s is not None else 1
                   self.B = np.array([])
        if np.allclose(s, 0.0):
            self.k = 0
            self.s = 0
            self.B = np.array([])

                









    
