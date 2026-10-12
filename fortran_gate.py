"""Bounded native Fortran closure, reports only; no source builds."""
import hashlib, json, os, shutil, subprocess, sys, sysconfig, tarfile, urllib.request
from pathlib import Path
import pefile
ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ['EVIDENCE']); OUT.mkdir(parents=True,exist_ok=True)
WORK=Path(os.environ['WORK']); WORK.mkdir(parents=True,exist_ok=True)
REPORT={'platform':sysconfig.get_platform(),'attempts':[],'green':False}
def save(): (OUT/'fortran-gate.json').write_text(json.dumps(REPORT,indent=2),encoding='utf8')
def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def run(cmd,tag,cwd=WORK,env=None,timeout=180):
    p=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,capture_output=True,timeout=timeout)
    text=(p.stdout+p.stderr).decode(errors='replace')
    (OUT/(tag+'.log')).write_text(text,encoding='utf8')
    result={'cmd':[str(x) for x in cmd],'returncode':p.returncode,'log':tag+'.log'}
    print(json.dumps(result),text[-5000:],flush=True)
    return result
def audit(p):
    pe=pefile.PE(str(p)); return {'path':str(p),'sha256':digest(p),'machine':hex(pe.FILE_HEADER.Machine),'imports':[d.dll.decode() for d in getattr(pe,'DIRECTORY_ENTRY_IMPORT',[])]}
def inventory(root):
    found=[]
    for base,dirs,files in os.walk(root):
        for name in files:
            low=name.lower()
            if low.endswith('.mod') or ('flang' in low or 'fortran' in low) and low.endswith(('.lib','.dll','.exe')):
                p=Path(base)/name; found.append({'path':str(p),'size':p.stat().st_size,'sha256':digest(p)})
    return found
assert sysconfig.get_platform()=='win-arm64'
REPORT['env']={k:os.environ.get(k) for k in ['VCToolsVersion','VCToolsInstallDir','LIB','INCLUDE','OPENBLAS_NUM_THREADS']}
REPORT['installed']=[]
roots=[Path(r'C:\Program Files\LLVM'),Path(os.environ['VSROOT'])/'VC/Tools/Llvm']
for root in roots: REPORT['installed'].extend(inventory(root))
save()
# Preserve exact public wheel manifest and hash, without uploading the binary.
spec=json.loads((ROOT/'blas.json').read_text())
f=WORK/spec['filename']; urllib.request.urlretrieve(spec['url'],f)
assert digest(f)==spec['digests']['sha256']; REPORT['blas_source']=spec
assert run([sys.executable,'-m','pip','install','--no-deps',f],'install-blas')['returncode']==0
import scipy_openblas32 as blas
libdir=Path(blas.get_lib_dir()); REPORT['blas_config']=blas.get_openblas_config()
assert 'USE64BITINT' not in REPORT['blas_config']
REPORT['blas_files']=[{'path':str(p),'size':p.stat().st_size,'sha256':digest(p)} for p in libdir.iterdir() if p.is_file()]
dll=libdir/'scipy_openblas.dll'; REPORT['blas_pe']=audit(dll); assert REPORT['blas_pe']['machine']=='0xaa64'
implibs=list(libdir.glob('*.lib')); assert len(implibs)==1,implibs
run(['dumpbin','/headers',implibs[0]],'blas-lib-headers'); run(['dumpbin','/exports',dll],'blas-exports')
shutil.copy2(dll,WORK/dll.name)
REPORT['c_compile']=run(['clang-cl','/nologo','/MD','/O2','/c',ROOT/'abi.c','/Fo'+str(WORK/'abi.obj')],'c-abi')
assert REPORT['c_compile']['returncode']==0

def attempt(fc,label,include=None):
    rec={'compiler':audit(fc),'label':label}; REPORT['attempts'].append(rec)
    rec['version']=run([fc,'--version'],label+'-version')
    exe=WORK/(label+'.exe')
    cmd=[fc,'-v','-O2','-fms-runtime-lib=dll']
    if include: cmd+=['-I'+str(include)]
    cmd += [ROOT/'interop.f90',WORK/'abi.obj',implibs[0],'-o',exe]
    rec['build']=run(cmd,label+'-build'); save()
    if rec['build']['returncode']!=0: return False
    rec['exe']=audit(exe); assert rec['exe']['machine']=='0xaa64'
    env=os.environ.copy(); env['PATH']=str(Path(fc).parent)+';'+str(libdir)+';'+env['PATH']
    rec['executions']=[]
    for threads in [1,2]:
        env['OPENBLAS_NUM_THREADS']=str(threads)
        result=run([exe],label+'-run-'+str(threads),env=env)
        result['OPENBLAS_NUM_THREADS']=threads; rec['executions'].append(result); save()
        if result['returncode']!=0 or 'FORTRAN_ISO_C_REAL_COMPLEX_LP64_OK' not in (OUT/result['log']).read_text(): return False
    rec['negative']=run([exe,'poison'],label+'-poison',env=env)
    assert rec['negative']['returncode']==91
    # Record recursive imported runtime closure. Never copy or publish CRT payloads.
    search=[WORK,Path(fc).parent,Path(fc).parent.parent/'lib',libdir,Path(os.environ['SystemRoot'])/'System32']+[Path(x) for x in env['PATH'].split(';') if x]
    pending=[exe,dll]; seen=set(); closure=[]
    while pending:
        binary=pending.pop(); key=str(binary).lower()
        if key in seen: continue
        seen.add(key); item=audit(binary); closure.append(item)
        for name in item['imports']:
            if name.lower().startswith(('api-ms-','ext-ms-')): continue
            target=next((d/name for d in search if (d/name).is_file()),None)
            if target is None: item.setdefault('unresolved',[]).append(name)
            elif str(target).lower().startswith(str(Path(os.environ['SystemRoot'])).lower()):
                item.setdefault('system_boundary',[]).append({'name':name,'path':str(target),'sha256':digest(target)})
            else: pending.append(target)
    rec['closure']=closure
    if any(x.get('unresolved') for x in closure): return False
    REPORT['green']=True; REPORT['selected_compiler']=str(fc); save(); return True

installed=shutil.which('flang') or shutil.which('flang-new')
if installed:
    if attempt(installed,'installed-default'): raise SystemExit(0)
    # Only use module files belonging to this exact installed distribution root.
    same=Path(installed).parent.parent
    mods=list(same.rglob('iso_c_binding.mod'))
    for i,m in enumerate(mods):
        if attempt(installed,'installed-explicit-'+str(i),m.parent): raise SystemExit(0)

# Full official portable archive, not NSIS installer and not a giant source build.
url='https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/clang%2Bllvm-22.1.8-aarch64-pc-windows-msvc.tar.xz'
sha='de718c58ebbc5f61d58c17b90457fcf42983bc2c4a4aba3e010d108713bfd7f1'
archive=WORK/'llvm22.1.8-arm64.tar.xz'; REPORT['official']={'url':url,'expected_sha256':sha}; save()
print('Downloading official portable LLVM archive',flush=True)
urllib.request.urlretrieve(url,archive); REPORT['official']['sha256']=digest(archive); assert digest(archive)==sha
unpack=WORK/'official'; unpack.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
    members=t.getmembers(); (OUT/'official-archive-members.txt').write_text('\n'.join(x.name for x in members),encoding='utf8')
    t.extractall(unpack,filter='data')
REPORT['official']['files']=inventory(unpack); save()
compilers=list(unpack.rglob('flang.exe'))
REPORT['official']['flang_candidates']=[str(x) for x in compilers]; save()
for i,fc in enumerate(compilers):
    if attempt(fc,'official-'+str(i)): raise SystemExit(0)
REPORT['frontier']='No verified complete native Fortran compiler/module/runtime closure; NumPy/SciPy builds not authorized across failed gate'
save(); raise SystemExit(1)
