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

    #@contract
    def __init__(self, vectors=None, GPUdevice=None, F=None, O=None, 
                 W=None, Mf=None):
        pass




