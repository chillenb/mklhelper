import numpy as np
import scipy
import pytest
from mklhelper import dtrsm, inspect
import scipy.linalg.blas as scipyblas


def test_trsm():
    rng = np.random.default_rng(0)
    m = 10
    n = 5

    M = rng.standard_normal((m, m))
    _, R = scipy.linalg.qr(M)
    A = R + np.eye(m)
    
    
    B = rng.standard_normal((m, n))

    X = np.zeros((m, n))
    X += B
    
    X_correct = scipy.linalg.solve_triangular(A, B, lower=False)
    
    dtrsm(A, X, uplo='U')
    
    assert np.allclose(X, X_correct)


    X = np.zeros((m, n))
    X += B
    
    A = np.asfortranarray(A)
    dtrsm(A, X, uplo='U')

    print(X)
    print(X_correct)
    assert np.allclose(X, X_correct)