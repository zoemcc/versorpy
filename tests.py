import numpy as np
import scipy.linalg as la
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

    # too many columns
    C = np.array([[1, 0, 1], [0, 1, 0]])
    B2 = Blade(C)
    assert B2.blade.shape == (1, 0)
    assert np.allclose(B2.s, 0)
    assert B2.k == 0


def testBladeInitColumnVector():
    # testing column vector
    c = np.array([[2], [0], [0]])
    B = Blade(c)
    print 's: ', B.s
    print 'blade: ', B.blade
    assert np.allclose(np.dot(B.blade.T, B.blade), np.eye(1))
    assert np.allclose(B.s, 2.0)
    assert B.k == 1

    B = Blade(c, s=2.0)
    print 's: ', B.s
    print 'blade: ', B.blade
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
        assert np.allclose(B.s, (-1)**(B.k * (B.k - 1) / 2) * bd.reverseScaling(B))
        assert B.k == i


def testBladeInvolutio():
    # sign
    for i in range(1, 5):
        B = Blade(np.eye(i))
        assert np.allclose(B.s, (-1)**B.k * bd.involution(B).s)
        assert np.allclose(B.s, (-1)**B.k * bd.involutionScaling(B))

def testBladeInnerSign():
    # sign
    for i in range(1, 5):
        B = Blade(np.eye(i, i))
        print 'B inner B: ', bd.inner(B, B).s
        assert np.allclose((-1)**((i * (i - 1)) / 2), bd.inner(B, B).s), 'value'
        assert bd.inner(B, B).blade.shape == (1, 0), 'shape'

def testBladeInnerScale():
    # scale
    A = np.array([[2, 0], [0, 1]])
    print 'A.T A : ', np.dot(A.T, A)
    
    B = Blade(A)
    print 'B inner B: ', bd.inner(B, B).s
    assert np.allclose(-4, bd.inner(B, B).s)

    A = np.array([[-2, 0], [0, 1]])
    print 'A.T A : ', np.dot(A.T, A)
    
    B = Blade(A)
    print 'B inner B: ', bd.inner(B, B).s
    assert np.allclose(-4, bd.inner(B, B).s)

def testBladeInnerNonGradeMatch():
    B1 = Blade(np.eye(2))
    B2 = Blade(np.eye(1))
    print 'B inner B: ', bd.inner(B1, B2).s
    assert np.allclose(0, bd.inner(B1, B2).s)

def testBladeInnerCommutativity():
    # commutative
    for i in range(1, 5):
        B1 = Blade(np.eye(i, i))
        B2 = Blade(-np.eye(i, i))
        firstOrdering = bd.inner(B1, B2)
        secondOrdering = bd.inner(B2, B1)
        assert np.allclose(firstOrdering.s, secondOrdering.s), (firstOrdering.s, secondOrdering.s)

def testBladeInverse():
    B1 = Blade(np.eye(2))
    print 'B1 inverse: ', bd.inverse(B1).s
    assert np.allclose(-B1.s, bd.inverse(B1).s)
    assert np.allclose(bd.inner(bd.inverse(B1), B1).s, 1.0)

    B2 = Blade(np.eye(1))
    print 'B2 inverse: ', bd.inverse(B2).s
    assert np.allclose(B2.s, bd.inverse(B2).s)
    assert np.allclose(bd.inner(bd.inverse(B2), B2).s, 1.0)

    B1 = Blade(np.eye(2), s=2.0)
    print 'B1 inverse: ', bd.inverse(B1).s, '-B1.s / 2.0: ', -B1.s / 4.0
    assert np.allclose(-B1.s / 4.0, bd.inverse(B1).s)
    assert np.allclose(bd.inner(bd.inverse(B1), B1).s, 1.0)

    B2 = Blade(np.eye(1), s=2.0)
    print 'B2 inverse: ', bd.inverse(B2).s
    assert np.allclose(B2.s / 4.0, bd.inverse(B2).s)
    assert np.allclose(bd.inner(bd.inverse(B2), B2).s, 1.0)

def testBladeCopy():
    B1 = Blade(np.eye(2))
    B2 = bd.copy(B1)
    assert np.allclose(B1.blade, B2.blade)
    assert np.allclose(B1.s, B2.s)

    B1.blade[1, 1] = 2
    assert not np.allclose(B1.blade, B2.blade)



def testBladeOuterSign():
    # sign/antisymmetric
    B1 = Blade(np.array([1, 0]))
    B2 = Blade(np.array([0, 1]))
    result1 = bd.outer(B1, B2)
    print 'B1 outer B2 s: ', result1.s
    print 'B1 outer B2 blade: ', result1.blade
    assert la.det(result1.blade) > 0, la.det(result1.blade)
    assert result1.blade.shape == (2, 2), result1.blade.shape
    assert np.allclose(np.eye(2), np.dot(result1.blade.T, result1.blade))
    assert np.allclose(bd.inner(result1, result1).s, -1), bd.inner(result1, result1).s

    result2 = bd.outer(B2, B1)
    print 'B2 outer B1 s: ', result2.s
    print 'B2 outer B1 blade: ', result2.blade
    assert la.det(result2.blade) < 0, la.det(result2.blade)
    assert result2.blade.shape == (2, 2), result2.blade.shape
    assert np.allclose(np.eye(2), np.dot(result2.blade.T, result2.blade))
    assert np.allclose(bd.inner(result2, result2).s, -1), bd.inner(result2, result2).s
    
    # result1 == inverse(result2)
    assert bd.inner(result1, result2).s > 0, bd.inner(result1, result2).s


def testBladeOuterScalar():
    # scalar to blade/ scalar to scalar
    B1 = Blade(np.eye(2))
    B2 = Blade(1, s=2)

    result1 = bd.outer(B1, B2)
    print 'B1 outer B2 s: ', result1.s
    assert np.allclose(2, result1.s), 'value'
    assert result1.blade.shape == (2, 2), 'shape'

    result2 = bd.outer(B2, B1)
    print 'B2 outer B1 s: ', result2.s
    assert np.allclose(2, result2.s), 'value'
    assert result2.blade.shape == (2, 2), 'shape'

    result3 = bd.outer(B2, B2)
    print 'B2 outer B2 s: ', result3.s
    assert np.allclose(result3.s, 4)


def testBladeOuterScaleMultplicative():
    # scale (just multiplicative scaling)
    B1 = Blade(np.array([2, 0]))
    B2 = Blade(np.array([0, 1]))
    result1 = bd.outer(B1, B2)
    print 'B1 outer B2 s: ', result1.s
    print 'B1 outer B2 blade: ', result1.blade
    assert la.det(result1.blade) > 0, la.det(result1.blade)
    assert result1.blade.shape == (2, 2), result1.blade.shape
    assert np.allclose(np.eye(2), np.dot(result1.blade.T, result1.blade))
    assert np.allclose(bd.inner(result1, result1).s, -4), bd.inner(result1, result1).s

    result2 = bd.outer(B2, B1)
    print 'B2 outer B1 s: ', result2.s
    print 'B2 outer B1 blade: ', result2.blade
    assert la.det(result2.blade) < 0, la.det(result2.blade)
    assert result2.blade.shape == (2, 2), result2.blade.shape
    assert np.allclose(np.eye(2), np.dot(result2.blade.T, result2.blade))
    assert np.allclose(bd.inner(result2, result2).s, -4), bd.inner(result2, result2).s
    
    # result1 == inverse(result2)
    assert np.allclose(bd.inner(result1, result2).s, 4), bd.inner(result1, result2).s

def testBladeOuterScaleParallelopiped():
    # scale (outer(B1, B2).s == 'area spanned by parallelopiped between B1, B2')
    B1 = Blade(np.array([1, 0]))
    B2 = Blade(np.array([1, 1])) # should have same area as [0, 1] when outered with B1
    result1 = bd.outer(B1, B2)
    print 'B1 outer B2 s: ', result1.s
    print 'B1 outer B2 blade: ', result1.blade
    assert la.det(result1.blade) > 0, la.det(result1.blade)
    assert result1.blade.shape == (2, 2), result1.blade.shape
    assert np.allclose(np.eye(2), np.dot(result1.blade.T, result1.blade))
    assert np.allclose(bd.inner(result1, result1).s, -1), bd.inner(result1, result1).s

    result2 = bd.outer(B2, B1)
    print 'B2 outer B1 s: ', result2.s
    print 'B2 outer B1 blade: ', result2.blade
    assert la.det(result2.blade) < 0, la.det(result2.blade)
    assert result2.blade.shape == (2, 2), result2.blade.shape
    assert np.allclose(np.eye(2), np.dot(result2.blade.T, result2.blade))
    assert np.allclose(bd.inner(result2, result2).s, -1), bd.inner(result2, result2).s
    
    # result1 == inverse(result2)
    assert np.allclose(bd.inner(result1, result2).s, 1), bd.inner(result1, result2).s

def testBladeOuterSameSubspace():
    B1 = Blade(np.eye(2))
    B2 = Blade(np.eye(2)[:, 0])
    result = bd.outer(B1, B2)
    print 'B1 outer B2: ', result.s
    assert np.allclose(0, result.s)

    B1 = Blade(np.eye(2))
    B2 = Blade(np.eye(2))
    result = bd.outer(B1, B2)
    print 'B1 outer B2: ', result.s
    assert np.allclose(0, result.s)





if __name__ == "__main__":
    nose.main()

