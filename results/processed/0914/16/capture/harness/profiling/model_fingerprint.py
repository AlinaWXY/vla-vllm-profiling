"""Fingerprint the actual loaded model, outside timed/profiled scopes."""
import hashlib
import json


def fingerprint(model):
    import torch
    digest=hashlib.sha256()
    count=0
    for name,tensor in sorted(model.named_parameters()):
        header=json.dumps([name,list(tensor.shape),str(tensor.dtype)],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'little'));digest.update(header)
        # uint8 view supports BF16, whose direct NumPy conversion is unavailable.
        value=tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(value))
        count+=1
    return {'sha256':digest.hexdigest(),'parameter_names':count,'method':'sorted names, shapes, dtypes and actual raw parameter bytes; outside timing'}
