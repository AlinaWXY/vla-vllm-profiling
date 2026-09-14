"""Replay a recorded Inductor launch policy without process-to-process retuning.

Only process-local launch selection is affected. Installed packages and model
sources are unchanged; every requested configuration must match the actual
post-warmup launch audit.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

_expected = None
_used = set()


def install(path):
    global _expected
    import triton
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    data = json.loads(Path(path).read_text())['compiled_launch_choices']
    if _expected is not None:
        raise RuntimeError('Launch policy already installed')
    _expected = data
    configs = {}
    for name, choices in data.items():
        if len(choices) != 1:
            raise ValueError(f'Expected one selected launch configuration: {name}')
        fields = {key.strip(): ast.literal_eval(value.strip())
                  for key, value in (part.split(':', 1) for part in choices[0].split(','))}
        options = {key: fields.pop(key) for key in ('num_warps', 'num_ctas', 'num_stages', 'maxnreg')}
        configs[name] = triton.Config(fields, **options)
    original = CachingAutotuner.run

    def run(self, *args, **kwargs):
        name = self.inductor_meta.get('kernel_name', self.fn.__name__)
        if name in configs and not getattr(self, '_pi05_launch_frozen', False):
            desired = configs[name]
            if not self.launchers:
                self.precompile()
            launcher = next((item for item in self.launchers if str(item.config)==str(desired)), None)
            if launcher is None:
                if self.fn.fn is None:
                    self.fn = self._reload_kernel().fn
                launcher = self._precompile_config(desired).make_launcher()
            # Coordinate-descent tuning can otherwise change the chosen launch
            # again on the first run in each freshly spawned application pass.
            launcher.config.found_by_coordesc = True
            self.launchers = [launcher]
            self._pi05_launch_frozen = True
            _used.add(name)
        return original(self, *args, **kwargs)

    CachingAutotuner.run = run


def audit(actual):
    if _expected is None:
        return None
    if actual != _expected or _used != set(_expected):
        raise ValueError(f'Actual compiled launch policy differs from recording; unused={set(_expected)-_used}')
    return {'selected_configurations':len(_expected), 'all_used':True, 'all_match_recording':True}
