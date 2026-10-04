"""Cross-compile CUDA source to CUBIN without requiring the target GPU.
Usage: python nvrtc-cubin.test.py SOURCE ENTRY [SM ...]
"""
import ctypes as c
import json
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
dll_dir=root/'build-portable'/'Release'
handle=os.add_dll_directory(str(dll_dir))
nv=c.CDLL(str(next(dll_dir.glob('nvrtc64_*.dll'))))
P=c.c_void_p; S=c.c_char_p; I=c.c_int; Z=c.c_size_t

def api(name,args):
    f=getattr(nv,name);f.argtypes=args;f.restype=I;return f

create=api('nvrtcCreateProgram',[c.POINTER(P),S,S,I,P,P])
name=api('nvrtcAddNameExpression',[P,S])
compile_=api('nvrtcCompileProgram',[P,I,c.POINTER(S)])
log_size=api('nvrtcGetProgramLogSize',[P,c.POINTER(Z)])
log_get=api('nvrtcGetProgramLog',[P,P])
size=api('nvrtcGetCUBINSize',[P,c.POINTER(Z)])
get=api('nvrtcGetCUBIN',[P,P])
destroy=api('nvrtcDestroyProgram',[c.POINTER(P)])

def check(result):
    if result:raise RuntimeError(f'NVRTC error {result}')

source=Path(sys.argv[1]).read_bytes();entry=sys.argv[2].encode()
for sm in sys.argv[3:] or ['86','120']:
    program=P();check(create(c.byref(program),source,b'test.cu',0,None,None))
    try:
        check(name(program,entry))
        options=[f'--gpu-architecture=sm_{sm}'.encode(),b'--std=c++17',b'--no-source-include',b'--use_fast_math',b'--dopt=on',b'--Ofast-compile=0',b'--extra-device-vectorization',b'--ptxas-options=--opt-level=3']
        result=compile_(program,len(options),(S*len(options))(*options))
        n=Z();check(log_size(program,c.byref(n)));log=c.create_string_buffer(n.value);check(log_get(program,log))
        if result:raise RuntimeError(log.value.decode())
        check(size(program,c.byref(n)));assert n.value>0
        data=c.create_string_buffer(n.value);check(get(program,data));assert data.raw.startswith(b'\x7fELF')
        print(json.dumps({'architecture':f'sm_{sm}','codeFormat':'cubin','bytes':n.value,'compiled':True,'executed':False,'log':log.value.decode()}))
    finally:check(destroy(c.byref(program)))
