"""Prepare disposable BF16 serialization off Thor; preserve original checkpoint.

Pi05Pipeline blanket-casts the loaded model to BF16. This cache reduces NFS
traffic, and must be checked against the original model's parameter fingerprint
before profiling. It does not replace the source checkpoint or its identity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.experiments import timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--record",type=Path,required=True)
    args=p.parse_args()
    if socket.gethostname().split('.')[0] in ('thor0','fact-thor'):
        raise RuntimeError("Prepare this disposable cache on another host")
    if not str(args.output.resolve()).startswith('/scratch/'):
        raise ValueError("Derived cache must remain disposable under /scratch")
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started=timestamp()
    path=args.checkpoint/'model.safetensors'
    with path.open('rb') as f:
        header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
    names=sorted(n for n in header if n!='__metadata__')
    output_header={'__metadata__':{'format':'pt','derived_from':'pi05_base F32; BF16 model-load cache'}}
    offset=0
    for name in names:
        spec=header[name]
        if spec['dtype']!='F32':
            raise ValueError(f"Unexpected source dtype: {name} {spec['dtype']}")
        size=(spec['data_offsets'][1]-spec['data_offsets'][0])//2
        output_header[name]={'dtype':'BF16','shape':spec['shape'],'data_offsets':[offset,offset+size]}
        offset+=size
    encoded=json.dumps(output_header,separators=(',',':')).encode()
    encoded+=b' '*(-len(encoded)%8)
    destination=args.output/'model.safetensors'
    digest=hashlib.sha256()
    tensors=[]
    with destination.open('xb') as f,safe_open(path,framework='pt',device='cpu') as original:
        def write(data):
            digest.update(data);f.write(data)
        write(struct.pack('<Q',len(encoded)));write(encoded)
        for i,name in enumerate(names):
            tensor=original.get_tensor(name).to(torch.bfloat16)
            raw=tensor.view(torch.uint16).numpy().tobytes()
            write(raw)
            tensors.append({'name':name,'shape':list(tensor.shape),'bf16_sha256':hashlib.sha256(raw).hexdigest()})
            if i%100==0:
                print(timestamp(),i+1,'/',len(names),flush=True)
        f.flush();os.fsync(f.fileno())
    (args.output/'config.json').write_bytes((args.checkpoint/'config.json').read_bytes())
    record={'started_at':started,'finished_at':timestamp(),'host':socket.gethostname(),
        'source_checkpoint':str(args.checkpoint),'source_identity':json.loads((ROOT/'sources.lock.json').read_text())['checkpoint'],
        'cache':str(destination),'cache_bytes':destination.stat().st_size,'cache_sha256':digest.hexdigest(),
        'dtype':'BF16','tensor_count':len(tensors),'tensors':tensors,
        'use_condition':'Actual loaded BF16 model fingerprint must equal original-checkpoint baseline before profiling'}
    args.record.write_text(json.dumps(record,indent=2)+'\n')
    (args.output/'cache_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(timestamp(),'Saved',destination,record['cache_bytes'],flush=True)


if __name__=='__main__':
    main()
