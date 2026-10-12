"""Bounded metadata preparation only; expected failures remain explicit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import urllib.request

out=Path(os.environ['EVIDENCE']); out.mkdir(parents=True,exist_ok=True)
assert sysconfig.get_platform()=='win-arm64' and sys.version_info[:2]==(3,12)
results=[]
for package,version in [('scipy','1.11.2'),('numpy','1.22.2'),('numpy','1.23.5')]:
    label=package+'-'+version
    venv=out.parent/('venv-'+label); subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
    py=str(venv/'Scripts/python.exe')
    # Hash-verify the exact public sdist before asking pip to prepare metadata.
    url=f'https://pypi.org/pypi/{package}/{version}/json'
    meta=json.load(urllib.request.urlopen(url)); f=next(x for x in meta['urls'] if x['packagetype']=='sdist')
    raw=urllib.request.urlopen(f['url']).read(); assert hashlib.sha256(raw).hexdigest()==f['digests']['sha256']
    source=out.parent/f['filename']; source.write_bytes(raw)
    cmd=[py,'-m','pip','install','--dry-run','--ignore-installed','--no-deps',str(source)]
    try:
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        code=r.returncode; text=r.stdout.decode(errors='replace')
    except subprocess.TimeoutExpired as e:
        code=124; text=(e.stdout or b'').decode(errors='replace')+'\nTIMEOUT\n'
    (out/(label+'.log')).write_text(text)
    results.append({'package':package,'version':version,'sdist':f,'requires_python':meta['info']['requires_python'],'command':cmd,'returncode':code,'metadata_prepared':code==0,'missing_numpy_1260b1':'No matching distribution found for numpy==1.26.0b1' in text,'impimporter_removed':'ImpImporter' in text,'timeout':code==124})
    (out/'metadata-results.json').write_text(json.dumps(results,indent=2))
    print(label,code,text[-2500:],flush=True)
# Success here means the audit ran, NOT that any wheel built or installed.
assert len(results)==3
assert all(x['returncode'] not in (0,124) for x in results), 'Unexpected success/timeout: inspect logs rather than infer a blocker'
assert results[0]['missing_numpy_1260b1'], 'SciPy failure not attributed to expected exact build pin'
assert all(x['impimporter_removed'] for x in results[1:]), 'Legacy NumPy failure not attributed to expected Python 3.12 backend'
