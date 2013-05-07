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

    @contract
    def __init__(self, p, GPUdevice=None, F=None, O=None, W=None, Mf=None, q=0):
        self.p = p
        self.q = q
        n = p + q
        self.n = n

        if GPUdevice is None:
            self.F = F if F is not None else np.zeros((n, n))
            self.O = O if O is not None else np.zeros((n, n))
            self.W = W if W is not None else np.zeros((n, n))
            self.Mf = Mf if Mf is not None else np.zeros((n, n))
        else: #gpuarray
            self.device = GPUdevice
            self.F = F if F is not None else np.zeros((n, n))
            self.O = O if O is not None else np.zeros((n, n))
            self.W = W if W is not None else np.zeros((n, n))
            self.Mf = Mf if Mf is not None else np.zeros((n, n))


