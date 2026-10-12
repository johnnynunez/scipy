"""Build pinned OpenBLAS, require numerical RED, apply one-line candidate, require GREEN."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'openblas-source'
BUILD = Path(os.environ['RUNNER_TEMP']) / 'gesv-source-build'
REPORT = Path(os.environ['EVIDENCE'])
REPORT.mkdir(parents=True, exist_ok=True)


def run(args, name, **kw):
    with (REPORT / (name + '.log')).open('w', encoding='utf8') as out:
        p = subprocess.run([str(x) for x in args], stdout=out, stderr=subprocess.STDOUT, **kw)
    print(name, p.returncode, flush=True)
    if p.returncode:
        print((REPORT / (name + '.log')).read_text(encoding='utf8')[-15000:], flush=True)
        raise SystemExit(p.returncode)


for exe in ('clang-cl', 'flang-new', 'cmake', 'ninja'):
    run([exe, '--version'], 'version-' + exe)
args = ['cmake', '-S', SOURCE, '-B', BUILD, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DTARGET=ARMV8', '-DBINARY=64',
        '-DCMAKE_C_COMPILER=clang-cl', '-DCMAKE_Fortran_COMPILER=flang-new',
        '-DBUILD_SHARED_LIBS=ON', '-DCMAKE_SYSTEM_PROCESSOR=arm64',
        '-DCMAKE_SYSTEM_NAME=Windows', '-DSYMBOLPREFIX=scipy_', '-DLIBNAMEPREFIX=scipy_',
        '-DUSE_THREADS=1', '-DNUM_THREADS=24', '-DBUILD_SINGLE=ON', '-DBUILD_DOUBLE=ON',
        '-DBUILD_COMPLEX=ON', '-DBUILD_COMPLEX16=ON', '-DBUILD_TESTING=OFF',
        '-DBUILD_WITHOUT_LAPACKE=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
(REPORT / 'build-profile.json').write_text(json.dumps({'source_commit': '9bdf051b96e956f848dfcef89c23e1993b0e1b3e',
    'configure': [str(x) for x in args], 'scope': 'all-precision LP64 library build, complex128 GESV regression; not a rebuilt NumPy wheel'}, indent=2))
run(args, 'configure')
run(['cmake', '--build', BUILD, '--parallel', '8'], 'build-red')
source = SOURCE / 'interface/lapack/gesv.c'
original = source.read_text()
assert original.count('args.nthreads = 4;') == 1
(REPORT / 'source-before.c.txt').write_text(original)
for phase in ('red', 'green'):
    if phase == 'green':
        source.write_text(original.replace('args.nthreads = 4;', 'args.nthreads = MIN(4, num_cpu_avail(4));'))
        run(['cmake', '--build', BUILD, '--parallel', '8'], 'build-green')
    dlls = list(BUILD.rglob('scipy_openblas.dll'))
    assert len(dlls) == 1, dlls
    dst = BUILD / ('preserved-' + phase)
    dst.mkdir()
    dll = dst / dlls[0].name
    shutil.copy2(dlls[0], dll)
    b = dll.read_bytes()
    pe = struct.unpack_from('<I', b, 0x3c)[0]
    machine = struct.unpack_from('<H', b, pe + 4)[0]
    assert machine == 0xaa64
    (REPORT / (phase + '-binary.json')).write_text(json.dumps({'dll_sha256': hashlib.sha256(b).hexdigest(),
        'pe_machine': hex(machine), 'source_gesv_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, indent=2))
    run([sys.executable, ROOT / 'gesv_rebuild_probe.py', dll, phase, REPORT / (phase + '-results.json')], 'test-' + phase)
(REPORT / 'completion.json').write_text(json.dumps({'source_red_reproduced': True, 'candidate_green': True,
    'scope': 'complex128-only native library; no complete wheel rebuild, full suite, application or GPU acceptance'}))
