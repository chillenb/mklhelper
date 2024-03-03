import numpy as np
import pytest
from mklhelper import dgemm, zgemm, inspect


def test_dgemm():
    rng = np.random.default_rng(0)
    m, k, n = 45, 20, 37

    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))
    C = np.zeros((m, n))

    dgemm(A, B, C)
    assert np.allclose(C, A @ B)

    A = np.asfortranarray(A)
    B = np.asfortranarray(B)

    C = np.zeros((m, n))
    
    dgemm(A, B, C)

    assert np.allclose(C, A @ B)

    A = np.ascontiguousarray(A)
    B = np.asfortranarray(B)

    C = np.zeros((m, n))
    
    dgemm(A, B, C)

    assert np.allclose(C, A @ B)

    A = np.asfortranarray(A)
    B = np.ascontiguousarray(B)

    C = np.zeros((m, n))
    
    dgemm(A, B, C)

    assert np.allclose(C, A @ B)


def test_zgemm():
    rng = np.random.default_rng(0)
    m, k, n = 45, 20, 37

    A = rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))
    B = rng.standard_normal((k, n)) + 1j * rng.standard_normal((k, n))
    C = np.zeros((m, n), dtype=complex)

    zgemm(A, B, C)
    assert np.allclose(C, A @ B)

    A = np.asfortranarray(A)
    B = np.asfortranarray(B)

    C = np.zeros((m, n), dtype=complex)
    
    zgemm(A, B, C)

    assert np.allclose(C, A @ B)

    A = np.ascontiguousarray(A)
    B = np.asfortranarray(B)

    C = np.zeros((m, n), dtype=complex)
    
    zgemm(A, B, C)

    assert np.allclose(C, A @ B)

    A = np.asfortranarray(A)
    B = np.ascontiguousarray(B)

    C = np.zeros((m, n), dtype=complex)
    
    zgemm(A, B, C)

    assert np.allclose(C, A @ B)