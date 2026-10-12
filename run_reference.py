"""Run already-published wheels, not a source build or old-pin substitute."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
from wheel_audit import audit_bytes

p=argparse.ArgumentParser(); p.add_argument('--version',required=True); p.add_argument('--out',required=True); a=p.parse_args()
out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
wheelhouse=out/'wheels'; wheelhouse.mkdir(exist_ok=True)
spec=json.loads(Path(__file__).with_name('wheels.json').read_text())
selected=[spec['numpy==2.5.1'],spec['scipy=='+a.version]]
audit={}
for f in selected:
    raw=urllib.request.urlopen(f['url'],timeout=60).read()
    assert hashlib.sha256(raw).hexdigest()==f['digests']['sha256']
    dest=wheelhouse/f['filename']; dest.write_bytes(raw)
    audit[dest.name]=audit_bytes(raw)
(out/'wheel-integrity.json').write_text(json.dumps(audit,indent=2))
(out/'published-provenance.json').write_text(json.dumps(selected,indent=2))
subprocess.run([sys.executable,'-m','pip','install',*[str(wheelhouse/f['filename']) for f in selected]],check=True)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
(out/'pip-freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze','--all']))
for n in ['test_oracles.py','runtime_inventory.py']:
    shutil.copy2(Path(__file__).with_name(n),out/n)
results=[]
def run(label,cmd,timeout=600,env=None,expected=0):
    try:
        r=subprocess.run(cmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        code=r.returncode; raw=r.stdout
    except subprocess.TimeoutExpired as exc:
        code=124; raw=(exc.stdout or b'')+b'\nHARNESS TIMEOUT\n'
    (out/(label+'.log')).write_bytes(raw)
    results.append({'label':label,'command':cmd,'returncode':code,'expected':expected,'passed':code==expected})
    (out/'commands.json').write_text(json.dumps(results,indent=2))
    print(label,code,raw.decode(errors='replace')[-2500:],flush=True)
run('oracle',[sys.executable,'-m','pytest','test_oracles.py','-q','--junitxml=oracle.xml','--timeout=60'])
env=dict(os.environ,POISON_ORACLE='1')
run('poison-negative',[sys.executable,'-m','pytest','test_oracles.py','-q','-k','test_linalg','--junitxml=poison-negative.xml','--timeout=60'],env=env,expected=1)
run('runtime-inventory',[sys.executable,'runtime_inventory.py'])
for scope in ['linalg','sparse','fft','optimize']:
    run('full-'+scope,[sys.executable,'-m','pytest','--pyargs','scipy.'+scope,'-q','-n','2','--junitxml=full-'+scope+'.xml','--timeout=90','--durations=15'],timeout=900)
raise SystemExit(0 if all(x['passed'] for x in results) else 1)
