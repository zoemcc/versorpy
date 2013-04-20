import numpy as np
import nose
from blade import Blade
# -------- #  Blade tests

# -------- #  Blade.__init__() tests

def testBladeInit():
    " test all possible permutations of inputting s and common things "
    B = Blade()
    assert np.allclose(B.B, np.array([]))

    A = np.array([[1, 0], [0, 1]])
    B = Blade(A, orthonormal=True)
    assert np.allclose(B.B, A)
    assert np.allclose(B.s, 1.0)

    B = Blade(A, orthonormal=True, s=2.0)
    assert np.allclose(B.B, A)
    assert np.allclose(B.s, 2.0)

    B = Blade(A, orthonormal=True, s=0.0)
    assert np.allclose(B.B, np.array([]))
    assert np.allclose(B.s, 0)

    B = Blade(A)
    assert np.allclose(B.B, A)
    assert np.allclose(B.s, 1)
