import numpy as np
import scipy.linalg as la
import math
import nose
from blade import Blade
import blade as bd
import numpy.random as rn
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
    B = Blade(1)
    assert np.allclose(B.s, (-1)**(B.k * (B.k - 1) / 2) * bd.reverse(B).s)
    assert np.allclose(B.s, (-1)**(B.k * (B.k - 1) / 2) * bd.reverseScaling(B))
    assert B.k == 0

    for i in range(1, 5):
        B = Blade(np.eye(i))
        assert np.allclose(B.s, (-1)**(B.k * (B.k - 1) / 2) * bd.reverse(B).s)
        assert np.allclose(B.s, (-1)**(B.k * (B.k - 1) / 2) * bd.reverseScaling(B))
        assert B.k == i


def testBladeInvolutio():
    # sign
    B = Blade(1)
    assert np.allclose(B.s, (-1)**B.k * bd.involution(B).s)
    assert np.allclose(B.s, (-1)**B.k * bd.involutionScaling(B))
    assert B.k == 0

    for i in range(1, 5):
        B = Blade(np.eye(i))
        assert np.allclose(B.s, (-1)**B.k * bd.involution(B).s)
        assert np.allclose(B.s, (-1)**B.k * bd.involutionScaling(B))
        assert B.k == i

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
    B = Blade(2)
    inv = bd.inverse(B)
    assert np.allclose(B.s * inv.s, 1.0), inv.s

    B = Blade(-2)
    inv = bd.inverse(B)
    assert np.allclose(B.s * inv.s, 1.0), inv.s

    B = Blade(0)
    try:
        inv = bd.inverse(B)
    except AttributeError as err:
        assert err[0] == 'Not invertible, s=0'
    except:
        assert False, err

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

    B1 = Blade(1, s=2)
    B2 = Blade(1, s=-.5)

    result1 = bd.outer(B1, B2)
    print 'B1 outer B2 s: ', result1.s
    assert np.allclose(-1, result1.s), 'value'
    assert result1.blade.shape == (1, 0), 'shape'



def testBladeOuterScaleMultplicative():
    # multiplicative scaling
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
    # test outer(B1, B2).s == 'area spanned by parallelopiped between B1, B2'
    B1 = Blade(np.array([1, 0]))
    B2 = Blade(np.array([1, 1])) # should have same area as [0, 1] when outered with B1
    result1 = bd.outer(B1, B2)
    print 'B1 outer B2 s: ', result1.s
    print 'B1 outer B2 blade: ', result1.blade
    assert la.det(result1.blade) > 0, la.det(result1.blade)
    assert result1.blade.shape == (2, 2), result1.blade.shape
    assert np.allclose(np.eye(2), np.dot(result1.blade.T, result1.blade))
    assert np.allclose(bd.inner(result1, result1).s, -1), bd.inner(result1, result1).s

    B1 = Blade(np.array([1, 0]), s=2.0)
    B2 = Blade(np.array([1, 1]), s=3.0) 
    result2 = bd.outer(B1, B2)
    print 'B1 outer B2 s: ', result2.s
    print 'B1 outer B2 blade: ', result2.blade
    assert la.det(result2.blade) > 0, la.det(result2.blade)
    assert result2.blade.shape == (2, 2), result2.blade.shape
    assert np.allclose(np.eye(2), np.dot(result2.blade.T, result2.blade))
    assert np.allclose(bd.inner(result2, result2).s, -36), bd.inner(result2, result2).s


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


def testBladeDualEuclidean():
    # s=0
    B = Blade(0)
    try:
        D = bd.dual(B)
    except AttributeError as err:
        assert err[0] == 'Not dualizable, s=0'
    except:
        assert False, err

    # scalar, expect a 2-blade
    A = Blade(2)
    D = bd.dual(A, n=2)
    revPseudo = bd.outer(bd.inverse(A), D)
    shouldBeOne = bd.inner(revPseudo, bd.pseudoScalar(2))
    assert np.allclose(shouldBeOne.s, 1.0), (shouldBeOne.blade, shouldBeOne.s)

    # 1-blade, expect 1-blade back
    A = Blade(np.array([1, 0]))
    D = bd.dual(A)
    revPseudo = bd.outer(bd.inverse(A), D)
    shouldBeOne = bd.inner(revPseudo, bd.pseudoScalar(2))
    assert np.allclose(shouldBeOne.s, 1.0), (shouldBeOne.blade, shouldBeOne.s)

    # 2-blade, expect 0-blade back
    A = Blade(np.eye(2))
    D = bd.dual(A)
    revPseudo = bd.outer(bd.inverse(A), D)
    shouldBeOne = bd.inner(revPseudo, bd.pseudoScalar(2))
    assert np.allclose(shouldBeOne.s, 1.0), (shouldBeOne.blade, shouldBeOne.s)


def testBladeDualEuclideanRegression():
    # regression test -- tests a bunch of functions and they all have to work 
    # for many input dimensions for this to succeed fully
    for n in range(1, 10):
        for k in range(0, n + 1):
            if k == 0:
                blade = Blade(1, s=float(rn.rand(1)[0]))
            else:
                blade = Blade(rn.randn(n, k))
            print 'blade n, k: ', blade.n, blade.k
            if np.allclose(blade.s, 0): # just skip this one for now... 
                continue                # measure 0 event but could happen
            D = bd.dual(blade, n=n)
            print 'D n, k: ', D.n, D.k
            revPseudo = bd.outer(bd.inverse(blade), D)
            print 'n, k: ', n, k
            print 'revPseudo blade, s: ', revPseudo.blade, revPseudo.s
            shouldBeOne = bd.inner(revPseudo, bd.pseudoScalar(n))
            assert np.allclose(shouldBeOne.s, 1.0), (shouldBeOne.blade, shouldBeOne.s)


def testBladeUnDualEuclidean():
    # s=0
    B = Blade(0)
    try:
        D = bd.undual(B)
    except AttributeError as err:
        assert err[0] == 'Not dualizable, s=0'
    except:
        assert False, err

    # scalar, expect a 2-blade
    A = Blade(2)
    Aprime = bd.dual(bd.undual(A, n=2), n=2)
    print 'A: ', A.blade, A.s
    print 'APrimeInv: ', bd.inverse(Aprime).blade, bd.inverse(Aprime).s
    shouldBeOne = bd.outer(A, bd.inverse(Aprime))
    print 'shouldBeOne: ', shouldBeOne.blade, shouldBeOne.s
    assert np.allclose(shouldBeOne.s, 1.0), (shouldBeOne.blade, shouldBeOne.s)

    # 1-blade, expect 1-blade back
    A = Blade(np.array([1, 0]))
    Aprime = bd.undual(bd.dual(A, n=2), n=2)
    shouldBeOne = bd.inner(A, bd.inverse(Aprime))
    assert np.allclose(shouldBeOne.s, 1.0), (shouldBeOne.blade, shouldBeOne.s)

    # 2-blade, expect 0-blade back
    A = Blade(np.eye(2))
    Aprime = bd.undual(bd.dual(A, n=2), n=2)
    print 'A: ', A.blade, A.s
    print 'APrimeInv: ', bd.inverse(Aprime).blade, bd.inverse(Aprime).s
    shouldBeOne = bd.inner(A, bd.inverse(Aprime))
    print 'shouldBeOne: ', shouldBeOne.blade, shouldBeOne.s
    assert np.allclose(shouldBeOne.s, 1.0), (shouldBeOne.blade, shouldBeOne.s)


def testBladeLeftContractBasic():
    # basic usage
    A = Blade(np.array([1, 0]))
    B = Blade(np.array([[1, 0], [0, 1]]))
    lc = bd.leftContract(A, B)
    assert np.allclose(lc.s, 1.0), (lc.blade, lc.s)
    assert (np.allclose(lc.blade, np.array([[0], [1]])) and \
                   np.allclose(lc.s, 1)) or \
           (np.allclose(lc.blade, np.array([[0], [-1]])) and \
                   np.allclose(lc.s, -1)), (lc.blade, lc.s)

def testBladeLeftContractOrthoProj():
    # orthogonal projection
    A = Blade(np.array([1, 1, 0]))
    B = Blade(np.array([[2, 0], [0, 0], [0, 2]]))
    Aproj = bd.leftContract(bd.leftContract(A, B), bd.inverse(B))
    assert (np.allclose(Aproj.blade, np.array([[1], [0], [0]])) and \
                   np.allclose(Aproj.s * la.norm(np.array([1, 1, 0])), A.s)) or \
           (np.allclose(-Aproj.blade, np.array([[-1], [0], [0]])) and \
                   np.allclose(-Aproj.s * la.norm(np.array([1, 1, 0])), A.s)) 

def testBladeJoinScalar():
    # scalars
    A = Blade(1, s=2)
    B = Blade(1, s=4)
    J = bd.join(A, B)
    assert J.blade.shape == (1, 0)
    assert J.s == 1

    # scalar/blade
    B = Blade(np.array([1, 0, 1]))
    J = bd.join(A, B)
    assert J.blade.shape == (3, 1)
    assert J.s == 1

    B = Blade(np.array([1, 0, 1]))
    J = bd.join(B, A)
    assert J.blade.shape == (3, 1)
    assert J.s == 1

def testBladeJoinBasic():
    # easy

    A = Blade(np.array([1, 0, 0]))
    B = Blade(np.array([0, 1, 0]))
    J = bd.join(A, B)
    assert J.blade.shape == (3, 2)
    assert J.s == 1





def testBladeMeet():
    # TODO: all of this
    pass

def testBladeEqualityAndSubEqualSelf():
    # self equal
    A = Blade(np.array([[1, 2], [2, 3], [0, 0]]))
    B = Blade(np.array([[1, 2], [2, 3], [0, 0]]))
    assert bd.subSpaceEquality(A, B)
    assert bd.equality(A, B)


    # self equal
    A = Blade(np.array([[1, 2], [2, 3], [0, 0]]))
    B = Blade(2 * np.array([[1, 2], [2, 3], [0, 0]]), s=.25)
    assert bd.subSpaceEquality(A, B)
    assert bd.equality(A, B)

def testBladeEqualityRotation():
    # 45-degree rotation in plane x-y
    A = Blade(np.array([[1, 0], [0, 1], [0, 0]]))
    B = Blade(np.array([[1, 1], [-1, 1], [0, 0]]), s=1.0 / math.sqrt(2))
    assert bd.subSpaceEquality(A, B)
    assert bd.equality(A, B)

def testBladeEqualityNotSameSubspace():
    # 45-degree rotation in plane x-y
    A = Blade(np.array([[1, 0, 0], [0, 1, 0]]))
    B = Blade(np.array([[1, 1, 1], [-1, 1, 1]]), s=1.0 / math.sqrt(3))
    print 'join dim: ', bd.join(A, B).k
    print 'equal: ', bd.equality(A, B)
    assert not bd.equality(A, B)
    assert bd.subSpaceEquality(A, B)

def testBladeLinearTransformRankDeficient():
    #rank deficient
    B = Blade(np.eye(2))
    T = np.array([[1, 0], [0, 0]])
    result = bd.applyLinearTransform(T, B)
    assert result.k == 0, (result.blade, result.s)

def testBladeLinearTransformUnit():
    # unit transform
    B = Blade(np.eye(2), s=2)
    T = np.array([[1, 0], [0, 1]])
    result = bd.applyLinearTransform(T, B)
    assert result.k == 2, (result.blade, result.s)
    assert result.s == 2, (result.blade, result.s)

def testBladeLinearTransformScale():
    # scale
    B = Blade(np.eye(2), s=2)
    T = np.array([[2, 0], [0, .75]])
    result = bd.applyLinearTransform(T, B)
    assert result.k == 2, (result.blade, result.s)
    assert result.s == 3, (result.blade, result.s)


if __name__ == "__main__":
    nose.main()

