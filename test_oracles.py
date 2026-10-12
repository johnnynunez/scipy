"""Independent CPU numerical oracles; never GPU evidence."""
import os
import platform
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy import fft, linalg, optimize, sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_info, threadpool_limits


def test_native_interpreter():
    assert sys.platform == 'win32'
    assert sys.version_info[:2] == (3, 12)
    assert sysconfig.get_platform() == 'win-arm64'
    assert platform.machine().lower() in ('arm64', 'aarch64')


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('threads', [1, 2])
def test_linalg(dtype, threads):
    tol = 2e-5 if np.dtype(dtype).itemsize <= 8 and dtype != np.float64 else 2e-12
    rng = np.random.default_rng(540)
    a = rng.normal(size=(17, 17)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a += 1j*rng.normal(size=a.shape).astype(dtype)
    a = a.conj().T @ a + np.eye(17, dtype=dtype)*3
    wanted = np.arange(17, dtype=dtype)/7
    b = a @ wanted
    if os.environ.get('POISON_ORACLE') == '1':
        wanted = wanted + 1
    with threadpool_limits(limits=threads, user_api='blas'):
        assert_allclose(np.linalg.solve(a, b), wanted, atol=tol, rtol=tol)
        assert_allclose(linalg.solve(a, b), wanted, atol=tol, rtol=tol)
        u, s, vh = linalg.svd(a)
        assert_allclose((u*s) @ vh, a, atol=tol*10, rtol=tol*10)
        vals, vecs = linalg.eigh(a)
        assert_allclose(a @ vecs, vecs*vals, atol=tol*10, rtol=tol*10)


def test_sparse():
    a = sparse.diags([-np.ones(40), np.full(41, 4.), -np.ones(40)], [-1, 0, 1], format='csc')
    expected = np.linspace(-2, 3, 41)
    b = a @ expected
    assert_allclose(spsolve(a, b), expected, rtol=1e-12, atol=1e-12)
    assert_allclose(a.tocsr() @ expected, b, rtol=1e-12)


def test_fft():
    n = 64
    x = np.cos(2*np.pi*7*np.arange(n)/n)
    y = fft.rfft(x)
    expected = np.zeros(33, dtype=complex)
    expected[7] = 32
    assert_allclose(y, expected, atol=5e-13)
    assert_allclose(fft.irfft(y), x, atol=1e-14)
    assert_allclose(np.fft.rfft(x), expected, atol=5e-13)


def test_optimize():
    result = optimize.minimize(lambda x: np.sum((x-np.array([2., -3.]))**2), [0., 0.], jac=lambda x: 2*(x-np.array([2., -3.])), method='BFGS')
    assert result.success, result.message
    assert_allclose(result.x, [2., -3.], atol=1e-9)
    linear = optimize.linprog([-1., -2.], A_ub=[[1., 1.]], b_ub=[3.], bounds=[(0, 2), (0, 2)], method='highs')
    assert linear.success, linear.message
    assert_allclose(linear.x, [1., 2.], atol=1e-10)
    assert_allclose(linear.fun, -5., atol=1e-10)


def test_concurrent_blas():
    def solve(seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(32,32)); a = x.T @ x + np.eye(32)
        b = np.arange(32.)
        assert_allclose(a @ linalg.solve(a,b), b, atol=1e-11)
    with threadpool_limits(limits=2):
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(solve, range(12)))
    assert any(x['user_api'] == 'blas' for x in threadpool_info())


def test_singular_negative():
    with pytest.raises(np.linalg.LinAlgError):
        linalg.solve(np.zeros((3,3)), np.ones(3))
