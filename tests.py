import numpy as np
import nose
from blade import Blade
import blade as bd
# -------- #  BEGIN Blade tests

# -------- #  SUBBEGIN Blade.__init__() tests

def testBladeInit():
    pass
def testBladeInitNull():
    " Testing null blade "
    B = Blade()
    assert B.blade.shape == (0,)
    assert np.allclose(B.s, 0)
    assert B.k == 0

    B = Blade(s=0)
    assert B.blade.shape == (0,)
    assert np.allclose(B.s, 0)
    assert B.k == 0

def testBladeInitScalar():
    " Testing scalar blade "
    B = Blade(1)
    assert B.blade.shape == (1, 1)
    assert np.allclose(B.s, 1)
    assert B.k == 1

    B = Blade(1, s=2)
    assert B.blade.shape == (1, 1)
    assert np.allclose(B.s, 2)
    assert B.k == 1

    B = Blade(2)
    assert B.blade.shape == (1, 1)
    assert np.allclose(B.s, 2)
    assert B.k == 1

    B = Blade(2, s=2)
    assert B.blade.shape == (1, 1)
    assert np.allclose(B.s, 4)
    assert B.k == 1

def testBladeInitOrthonormalQr():
    " Testing orthonormal and qr "
    # testing orthonormal
    A = np.array([[1, 0], [0, 1], [0, 0]])

    B = Blade(A, orthonormal=True)
    assert np.allclose(B.blade, A)
    assert np.allclose(B.s, 1.0)

    B = Blade(A, orthonormal=True, s=2.0)
    assert np.allclose(B.blade, A)
    assert np.allclose(B.s, 2.0)

    B = Blade(A, orthonormal=True, s=0.0)
    assert B.blade.shape == (0,)
    assert np.allclose(B.s, 0)
    assert B.k == 0

    # testing qr
    B = Blade(A)
    print 'B.blade: ', B.blade
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(2))
    assert np.allclose(B.s, 1)

    B = Blade(A, s=2.0)
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(2))
    assert np.allclose(B.s, 2.0)

def testBladeInitQrScaling():
    # testing qr scaling
    A = np.array([[2, 0], [0, 1], [0, 0]])

    B = Blade(A)
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(2))
    assert np.allclose(B.s, 2.0)

    B = Blade(A, s=2.0)
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(2))
    assert np.allclose(B.s, 4.0)

def testBladeInitRankDeficient():
    # testing rank deficient
    A = np.array([[1, 0], [1, 0], [0, 0]])

    B = Blade(A)
    assert B.blade.shape == (0,)
    assert np.allclose(B.s, 0)
    assert B.k == 0

    B = Blade(A, s=2.0)
    assert B.blade.shape == (0,)
    assert np.allclose(B.s, 0)
    assert B.k == 0

def testBladeInitColumnVector():
    # testing column vector
    c = np.array([[2], [0], [0]])

    B = Blade(c)
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(1))
    assert np.allclose(B.s, 2.0)
    assert B.k == 1

    B = Blade(c, s=2.0)
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(1))
    assert np.allclose(B.s, 4.0)
    assert B.k == 1


def testBladeInitOneIndexVector():
    # testing one-index (row-ish) vector
    c = np.array([2, 0, 0])

    B = Blade(c)
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(1))
    assert np.allclose(B.s, 2.0)
    assert B.k == 1

def testBladeReverse():
    for i in range(1, 5):
        B = Blade(np.eye(i))
        assert np.allclose(B.s, (-1)**(B.k * (B.k - 1) / 2) * bd.reverse(B).s)
        assert B.k == i


def testBladeInvolutio():
    for i in range(1, 5):
        B = Blade(np.eye(i))
        assert np.allclose(B.s, (-1)**B.k * bd.involution(B).s)
        assert B.k == i

def testBladeInnerProduct():
    for i in range(1, 5):
        B = Blade(np.eye(i + 2, i))
        assert np.allclose(B.s, (-1)**B.k * bd.involution(B).s)
        assert B.k == i
    


if __name__ == "__main__":
    nose.main()

