"""Inventory installed PEs/imports and loaded DLLs after CPU operations."""
import ctypes
from ctypes import wintypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import numpy as np
import scipy
from scipy import linalg, fft, sparse, optimize
from scipy.sparse.linalg import spsolve
import pefile
from threadpoolctl import threadpool_info

assert sysconfig.get_platform()=='win-arm64'
assert sys.version_info[:2]==(3,12)
np.linalg.solve(np.eye(3),np.ones(3)); linalg.svd(np.eye(3))
spsolve(sparse.eye(3,format='csc'),np.ones(3)); fft.fft(np.arange(8.))
optimize.linprog([1.,2.],bounds=[(0,1),(0,1)],method='highs')
root=Path(np.__file__).parent.parent
payloads=[]
for pkg in ['numpy','scipy','numpy.libs','scipy.libs']:
    for f in (root/pkg).rglob('*'):
        if f.suffix.lower() not in ('.pyd','.dll'): continue
        pe=pefile.PE(str(f))
        payloads.append({'path':str(f),'machine':hex(pe.FILE_HEADER.Machine),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(), 'imports':[x.dll.decode() for x in getattr(pe,'DIRECTORY_ENTRY_IMPORT',[])], 'delay_imports':[x.dll.decode() for x in getattr(pe,'DIRECTORY_ENTRY_DELAY_IMPORT',[])]})
# Enumerate actual mapped modules using pointer-sized Win32 handles.
k=ctypes.WinDLL('kernel32',use_last_error=True); ps=ctypes.WinDLL('psapi',use_last_error=True)
k.GetCurrentProcess.restype=wintypes.HANDLE
ps.EnumProcessModules.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.HMODULE),wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
ps.GetModuleFileNameExW.argtypes=[wintypes.HANDLE,wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
modules=(wintypes.HMODULE*4096)(); needed=wintypes.DWORD(); process=k.GetCurrentProcess()
assert ps.EnumProcessModules(process,modules,ctypes.sizeof(modules),ctypes.byref(needed))
assert needed.value<=ctypes.sizeof(modules)
loaded=[]
for i in range(needed.value//ctypes.sizeof(wintypes.HMODULE)):
    text=ctypes.create_unicode_buffer(32768)
    assert ps.GetModuleFileNameExW(process,modules[i],text,len(text))
    f=Path(text.value); pe=pefile.PE(str(f),fast_load=True)
    loaded.append({'path':str(f),'machine':hex(pe.FILE_HEADER.Machine)})
known={Path(x['path']).name.lower():x for x in payloads+loaded}
imports=[]
for f in payloads:
    for name in f['imports']+f['delay_imports']:
        low=name.lower()
        if low.startswith(('api-ms-','ext-ms-')): status='windows-api-set'
        elif low in known: status='payload-or-loaded'
        elif (Path(os.environ['SystemRoot'])/'System32'/name).is_file(): status='windows-system32-present'
        else: status='UNRESOLVED'
        imports.append({'consumer':f['path'],'import':name,'status':status})
report={'numpy':np.__version__,'scipy':scipy.__version__,'executable':sys.executable,'platform':sysconfig.get_platform(),'blas':threadpool_info(),'payloads':payloads,'loaded_modules':loaded,'import_edges':imports,'limits':'Hosted CPU only; API-set forwarding delegated to Windows loader; all four target subpackages exercised, optional unimported code not certified.'}
Path('runtime-inventory.json').write_text(json.dumps(report,indent=2))
np.show_config(); scipy.show_config(); print(json.dumps(report['blas'],indent=2))
assert all(x['machine']=='0xaa64' for x in payloads)
assert not [x for x in imports if x['status']=='UNRESOLVED']
