"""Numerical GESV test of an explicitly selected source-built DLL."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def child(dllpath, n, limit):
    import numpy as np
    assert sys.platform == 'win32' and platform.machine().lower() == 'arm64'
    assert sys.version_info[:2] == (3, 12)
    dll = c.CDLL(str(Path(dllpath).resolve()))
    config = dll.scipy_openblas_get_config
    config.restype = c.c_char_p
    get = dll.scipy_openblas_get_num_threads
    get.restype = c.c_int
    assert get() == limit
    assert 'USE64BITINT' not in config().decode()
    print(json.dumps({'phase': 'before', 'n': n, 'limit': limit,
        'configured': get(), 'config': config().decode(), 'dll': dllpath,
        'sha256': hashlib.sha256(Path(dllpath).read_bytes()).hexdigest()}), flush=True)
    rng = np.random.default_rng(982345982439826)
    b = rng.random(n)
    a = rng.random((n, n))
    b = b + rng.random(n) * 1j
    a = a + rng.random((n, n)) * 1j
    a = np.asfortranarray(np.triu(np.tril(a, 2), -1))
    aa, x = a.copy(order='F'), b.copy()
    piv = np.empty(n, dtype=np.int32)
    ptrint = c.POINTER(c.c_int)
    gesv = dll.scipy_zgesv_
    gesv.argtypes = [ptrint, ptrint, c.c_void_p, ptrint, ptrint, c.c_void_p, ptrint, ptrint]
    gesv.restype = None
    nn, nrhs, lda, ldb, info = (c.c_int(v) for v in (n, 1, n, n, -999))
    gesv(c.byref(nn), c.byref(nrhs), aa.ctypes.data, c.byref(lda),
         piv.ctypes.data_as(ptrint), x.ctypes.data, c.byref(ldb), c.byref(info))
    assert info.value == 0
    np.testing.assert_allclose(a @ x, b, rtol=1e-12, atol=1e-12)
    assert np.isfinite(x).all()
    assert get() == limit
    print(json.dumps({'phase': 'after', 'pass': True,
        'max_absolute_residual': float(np.max(np.abs(a @ x - b)))}), flush=True)


def main():
    if sys.argv[1] == 'child':
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        return
    dllpath, phase, out = sys.argv[1:]
    rows = []
    for limit in (1, 2, 4):
        for n in (2, 17, 18, 20, 31, 32):
            env = dict(os.environ, OPENBLAS_NUM_THREADS=str(limit), OMP_NUM_THREADS=str(limit))
            args = [sys.executable, str(Path(__file__).resolve()), 'child', dllpath, str(n), str(limit)]
            try:
                p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
                row = dict(phase=phase, limit=limit, n=n, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
            except subprocess.TimeoutExpired as exc:
                row = dict(phase=phase, limit=limit, n=n, timeout=True, stdout=str(exc.stdout), stderr=str(exc.stderr))
            rows.append(row)
            Path(out).write_text(json.dumps(rows, indent=2), encoding='utf8')
            print(json.dumps(row), flush=True)
    assert len(rows) == 18
    failed = [r for r in rows if r.get('returncode') != 0]
    if phase == 'red':
        # Refuse to apply the candidate unless unchanged source reproduces both frontiers.
        assert any(r['n'] == 20 and r['limit'] == 1 for r in failed)
        assert any(r['n'] == 31 and r['limit'] == 2 for r in failed)
    else:
        assert not failed, failed


if __name__ == '__main__':
    main()
