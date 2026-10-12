"""Separate NumPy reference crash from SciPy and thread control detection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

out=Path(os.environ['EVIDENCE']); work=out.parent/'current-diagnostic'; work.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','venv',str(work/'venv')],check=True)
py=str(work/'venv/Scripts/python.exe')
spec=json.loads(Path(__file__).with_name('wheels.json').read_text()); wheels=[]
for k in ['numpy==2.5.1','scipy==1.17.0']:
 f=spec[k]; raw=urllib.request.urlopen(f['url']).read(); assert hashlib.sha256(raw).hexdigest()==f['digests']['sha256']; dest=work/f['filename']; dest.write_bytes(raw); wheels.append(str(dest))
subprocess.run([py,'-m','pip','install',*wheels,'threadpoolctl==3.7.0'],check=True)
setup='''import numpy as np
import sys
rng=np.random.default_rng(982345982439826)
n=20
b=rng.random(n)
A=rng.random((n,n))
b=b+rng.random(n)*1j
A=A+rng.random((n,n))*1j
A=np.asfortranarray(np.triu(np.tril(A,2),-1))
print(np.__version__,A.dtype,A.shape,flush=True)
'''
results=[]
for name,code in [('numpy-only',setup+'''x=np.linalg.solve(A,b)
np.testing.assert_allclose(A@x,b,rtol=1e-12,atol=1e-12)
print("NUMPY_SOLVE_OK",flush=True)
'''),('scipy-only',setup+'''from scipy.linalg import solve
x=solve(A,b,assume_a="banded")
np.testing.assert_allclose(A@x,b,rtol=1e-12,atol=1e-12)
print("SCIPY_SOLVE_OK",flush=True)
'''),('direct-thread-control','''import ctypes as c
import json
from pathlib import Path
import numpy, scipy
from scipy import linalg
from threadpoolctl import threadpool_info
linalg.solve([[2.,0.],[0.,3.]],[2.,3.])
print("threadpoolctl",threadpool_info())
report=[]
root=Path(numpy.__file__).parent.parent
for package in ["numpy.libs","scipy.libs"]:
 for f in (root/package).glob("*openblas*.dll"):
  dll=c.CDLL(str(f)); found={}
  for suffix in ["", "64_"]:
   try:
    get=getattr(dll,"scipy_openblas_get_num_threads"+suffix); get.restype=c.c_int
    put=getattr(dll,"scipy_openblas_set_num_threads"+suffix); put.argtypes=[c.c_int]
    before=get(); observed=[]
    for n in [1,2]: put(n); observed.append(get())
    put(before)
    found={"before":before,"set_1_2":observed,"suffix":suffix}; break
   except AttributeError: pass
  report.append({"path":str(f),"control":found})
print(json.dumps(report,indent=2))
assert report and all(x["control"].get("set_1_2")==[1,2] for x in report)
''')]:
 for threads in (1, 2, 4):
  env=dict(os.environ,OPENBLAS_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads))
  r=subprocess.run([py,'-c',code],capture_output=True,text=True,timeout=120,cwd=work,env=env)
  (out/(name+'-threads'+str(threads)+'.log')).write_text(r.stdout+r.stderr)
  results.append({'name':name,'threads_env':threads,'source':code,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
(out/'current-diagnostic.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
# Diagnostic collector only; no replacement of original failing target suites.
