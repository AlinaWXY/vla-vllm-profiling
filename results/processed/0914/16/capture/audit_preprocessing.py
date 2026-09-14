"""CPU-only reconstruction through the original Omni preprocessing source."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from profiling.experiments import file_record, timestamp

torch.set_num_threads(1)
out=Path(__file__).resolve().parent
source=Path('/fact_data/xinyaowang/vllm-omni-pi05-numerical-fixes/vllm_omni/diffusion/models/pi05/processor_pi05.py')
spec=importlib.util.spec_from_file_location('original_omni_processor',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assets=Path('/scratch/xinyaowang/vla-vllm-profiling/assets')
config=json.loads((assets/'pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/config.json').read_text())
keys=[k for k in config['input_features'] if k.startswith('observation.images.')]
cfg=SimpleNamespace(**config,image_feature_keys=keys,image_key_map={},state_norm_stats=None,state_num_bins=256)
rng=np.random.default_rng(17)
obs={'prompt':'pick up the red block and place it in the bin','state':np.zeros(32,dtype=np.float32),
     'images':{k:rng.integers(0,256,(224,224,3),dtype=np.uint8) for k in keys}}
tok=AutoTokenizer.from_pretrained(str(assets/'paligemma_tokenizer'),local_files_only=True)
images,masks,tokens,token_masks,metadata=module.build_model_inputs(obs,cfg,tok,torch.device('cpu'),max_cameras=3,return_metadata=True)
def record(t):
    a=t.contiguous().view(torch.uint8).numpy()
    return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(memoryview(a)).hexdigest()}
reconstructed={'images':{k.rsplit('.',1)[-1]:record(t) for k,t in zip(keys,images)},
               'tokens':record(tokens),'token_masks':record(token_masks),
               'camera_masks':[bool(x.item()) for x in masks], 'prefix_len':metadata['prefix_valid_len']}
sg=json.loads((out/'baseline/report.json').read_text())['input_audit']
checks={k:sg[k]==v for k,v in reconstructed.items()}
audit={'audited_at':timestamp(),'device':'cpu','source':file_record(source),'reconstructed_omni_inputs':reconstructed,
       'sglang_reference':file_record(out/'baseline/report.json'),'exact_matches':checks}
(out/'preprocessing_comparison.json').write_text(json.dumps(audit,indent=2)+'\n')
assert all(checks.values()),checks
print(checks)
