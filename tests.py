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
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 0)
    assert B.k == 0

    B = Blade(s=0)
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 0)
    assert B.k == 0

def testBladeInitScalar():
    " Testing scalar blade "
    B = Blade(1)
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 1)
    assert B.k == 0

    B = Blade(1, s=2)
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 2)
    assert B.k == 0

    B = Blade(2)
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 2)
    assert B.k == 0

    B = Blade(2, s=2)
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 4)
    assert B.k == 0

    B = Blade(1, s=0)
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 0)
    assert B.k == 0

def testBladeInitOrthonormal():
    " Testing orthonormal "
    # testing orthonormal
    A = np.array([[1, 0], [0, 1], [0, 0]])
    B = Blade(A, orthonormal=True)
    assert np.allclose(B.blade, A)
    assert np.allclose(B.s, 1.0)

    B = Blade(A, orthonormal=True, s=2.0)
    assert np.allclose(B.blade, A)
    assert np.allclose(B.s, 2.0)

    B = Blade(A, orthonormal=True, s=0.0)
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 0)
    assert B.k == 0

def testBladeInitQr():
    " Testing qr "
    # testing qr
    A = np.array([[1, 0], [0, 1], [0, 0]])
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
    print B.blade.shape
    assert B.blade.shape == (1, 0)
    assert np.allclose(B.s, 0)
    assert B.k == 0

    B = Blade(A, s=2.0)
    assert B.blade.shape == (1, 0)
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
    # sign
    for i in range(1, 5):
        B = Blade(np.eye(i))
        assert np.allclose(B.s, (-1)**(B.k * (B.k - 1) / 2) * bd.reverse(B).s)
        assert B.k == i


def testBladeInvolutio():
    # sign
    for i in range(1, 5):
        B = Blade(np.eye(i))
        assert np.allclose(B.s, (-1)**B.k * bd.involution(B).s)
        assert B.k == i

def testBladeInnerProductSign():
    # sign
    for i in range(1, 5):
        B = Blade(np.eye(i, i))
        print 'B inner B: ', bd.inner(B, B).s
        assert np.allclose((-1)**((i * (i - 1)) / 2), bd.inner(B, B).s)
        assert B.k == i

def testBladeInnerProductScale():
    # scale
    A = np.array([[2, 0], [0, 1]])
    print 'A.T A : ', np.dot(A.T, A)
    
    B = Blade(A)
    print 'B inner B: ', bd.inner(B, B).s
    assert np.allclose(-4, bd.inner(B, B).s)

def testBladeInnerProductNonGradeMatch():
    B1 = Blade(np.eye(2))
    B2 = Blade(np.eye(1))
    print 'B inner B: ', bd.inner(B1, B2).s
    assert np.allclose(0, bd.inner(B1, B2).s)



if __name__ == "__main__":
    nose.main()

