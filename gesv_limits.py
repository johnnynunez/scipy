"""Reports-only native GESV numerical regression; failures are never normalized."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

EXPECTED_NUMPY_BLAS = 'd99d8897eb24c4c8ba6b0536d58ffb87bcccdf043b579bae8edce179da3f33fe'


def child(route, n, limit):
    import numpy as np
    assert sys.platform == 'win32' and platform.machine().lower() == 'arm64'
    assert sys.version_info[:2] == (3, 12)
    if route.startswith('scipy'):
        import scipy
        from scipy import linalg
        package = scipy
    else:
        package = np
    provider = next((Path(package.__file__).parent.parent / (package.__name__ + '.libs')).glob('*openblas*.dll'))
    digest = hashlib.sha256(provider.read_bytes()).hexdigest()
    if package is np:
        assert digest == EXPECTED_NUMPY_BLAS
    dll = c.CDLL(str(provider))
    get = dll.scipy_openblas_get_num_threads
    get.restype = c.c_int
    config = dll.scipy_openblas_get_config
    config.restype = c.c_char_p
    configured = get()
    print(json.dumps({'phase': 'before', 'route': route, 'n': n, 'limit': limit,
        'configured': configured, 'provider': str(provider), 'sha256': digest,
        'config': config().decode(), 'numpy': np.__version__,
        'package_version': package.__version__}), flush=True)
    assert configured == limit
    assert 'USE64BITINT' not in config().decode(), 'These fixtures require verified LP64'
    rng = np.random.default_rng(982345982439826)
    b = rng.random(n)
    a = rng.random((n, n))
    b = b + rng.random(n) * 1j
    a = a + rng.random((n, n)) * 1j
    a = np.asfortranarray(np.triu(np.tril(a, 2), -1))
    if route == 'numpy':
        x = np.linalg.solve(a, b)
    elif route == 'scipy-gen':
        x = linalg.solve(a, b, assume_a='gen')
    elif route == 'scipy-gesv':
        gesv = linalg.get_lapack_funcs('gesv', (a, b))
        assert gesv.typecode == 'z' and gesv.int_dtype.itemsize == 4
        lu, piv, x, info = gesv(a.copy(order='F'), b.copy())
        assert info == 0
    elif route == 'numpy-direct-gesv':
        # Public Fortran LAPACK ABI, LP64 interleaved complex128.
        gesv = dll.scipy_zgesv_
        ptrint = c.POINTER(c.c_int)
        gesv.argtypes = [ptrint, ptrint, c.c_void_p, ptrint, ptrint, c.c_void_p, ptrint, ptrint]
        gesv.restype = None
        aa = a.copy(order='F')
        x = b.copy()
        piv = np.empty(n, dtype=np.int32)
        nn, nrhs, lda, ldb, info = (c.c_int(v) for v in (n, 1, n, n, -999))
        gesv(c.byref(nn), c.byref(nrhs), aa.ctypes.data, c.byref(lda),
             piv.ctypes.data_as(ptrint), x.ctypes.data, c.byref(ldb), c.byref(info))
        assert info.value == 0
    else:
        raise ValueError(route)
    # Original residual acceptance and precision are unchanged.
    np.testing.assert_allclose(a @ x, b, rtol=1e-12, atol=1e-12)
    assert np.isfinite(x).all()
    print(json.dumps({'phase': 'after', 'configured': get(),
        'max_absolute_residual': float(np.max(np.abs(a @ x - b))), 'pass': True}), flush=True)
    assert get() == limit


def main():
    if len(sys.argv) > 1:
        child(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
        return
    evidence = Path(os.environ['EVIDENCE'])
    evidence.mkdir(parents=True, exist_ok=True)
    rows = []
    for limit in (1, 2, 4):
        for route in ('numpy', 'scipy-gen', 'scipy-gesv', 'numpy-direct-gesv'):
            sizes = (2, 17, 18, 20, 31, 32) if route in ('numpy-direct-gesv', 'scipy-gesv') else (20,)
            for n in sizes:
                env = dict(os.environ, OPENBLAS_NUM_THREADS=str(limit), OMP_NUM_THREADS=str(limit))
                args = [sys.executable, str(Path(__file__).resolve()), route, str(n), str(limit)]
                try:
                    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=20)
                    row = dict(route=route, n=n, limit=limit, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
                except subprocess.TimeoutExpired as exc:
                    row = dict(route=route, n=n, limit=limit, timeout=True,
                               stdout=str(exc.stdout), stderr=str(exc.stderr))
                rows.append(row)
                (evidence / 'gesv-results.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
                print(json.dumps(row), flush=True)
    assert len(rows) == 42
    failed = [r for r in rows if r.get('returncode') != 0]
    summary = {'total': len(rows), 'passed': len(rows) - len(failed), 'failed': len(failed),
               'os': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
               'scope': 'public library numerical tests; no source correction or full application acceptance'}
    (evidence / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
