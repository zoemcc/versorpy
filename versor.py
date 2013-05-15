import numpy as np
from pycuda import gpuarray
from contracts import contract

"""
    A versor class and operations on versors in n-dimensional spaces.
    This library is adapted from Daniel Fontijne's PhD thesis,
    "Efficient Implementation of Geometric Algebra"
    staff.science.uva.nl/~fontijne/phd.html
    referenced here as DFPhD
"""

class Versor(object):

    def __init__(self, vectors=None, lazyInit=False, copy=True, factors=None,
            orthoFac=None, W=None, Mf=None):
        """
            Creates a new versor.  This is a list of vectors (the factors)
            that are all multiplied together  using the geometric product.
            We store them from left to right, so that the order 
            of the product corresponds to the order of the vectors 
            in the matrix representation.

            It stores the k vector factors (that are n-dimensional)
            in a n x k matrix, but also 
            a few other supporting data structures to assist in 
            versor reduction. Versor reduction is 
            the process by which a list of k
            vectors in a versor factor are refactored to at maximum 
            number of n factors.  Reduction can reduce the number 
            of factors, and specifically it reduces a matrix L
            of k factors to a matrix of rank(L) factors.
            
            We support lazy multiplies, where we have a list of vectors to 
            be multiplied but have not yet been added to the auxilliary 
            data structures.
            To implement this functionality, we store a deque for each 
            of premultiplies and postmultiplies,
            which correspond to vectors to be added to the structure 
            from the left and from the right.
        

            :param vectors: A list or matrix consisting of the factors of the versor.
            :type vectors: array[NxK]|array[N]|None

            :param orthonormal: Is the input blade already orthonormal?
            :type orthonormal: bool|None

            :param s: A number to scale the result by.
            :type s: number|None

            :param gpu: Should the blade be stored on the gpu and computations
                        involving the blade be performed on the gpu?
            :type gpu: bool

            :param tol: Any number smaller than tol is treated as 0 for rank
                        calculations
            :type tol: float

            :param copy: Should we copy the input array instead of storing a pointer?
            :type copy: bool
        """

        
        if vectors is None and factors is None:   # no input versor, init to empty
            self.factors = np.array([[]])
            self.orthoFac   = np.array([[]])
            self.toOrthoBlade = np.array([[]])
            self.metricFactors = np.array([[]])
            self.
            self.n = 1
            self.k = 0
        else:
            pass
            


            

        if lazyInit:
            if vectors is None and factors is None:   # no input versor, init to empty
                pass


            

        
        




