"""Run the second authorized experiment after the first NCU process exits."""
import json
from pathlib import Path
import subprocess
import time
root=Path('/fact_data/xinyaowang/vla_vllm_profiling')
first=root/'results/processed/0913/01/ncu/collection.json'
while True:
    try:
        state=json.loads(first.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        state={}
    if state.get('status')=='failed':
        raise RuntimeError('First capture failed; second capture not started')
    if state.get('status')=='captured':
        break
    time.sleep(15)
with (root/'results/processed/0913/02/launcher.log').open('x') as log:
    subprocess.run(['bash','results/processed/0913/02/run.sh'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
