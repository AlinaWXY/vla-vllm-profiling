"""Label actual optimized Triton launches without changing their arguments.

Used only in the diagnostic, non-graph replay after measuring production graphs.
The original PR has no operator NVTX labels inside its optimized decoder.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect


@contextmanager
def observe_decoder(module):
    observed = {}
    cls = module.Pi05RealtimeTritonDecoder
    original = cls.__call__

    def wrapped(self, **kwargs):
        observed["decoder"] = self
        observed["kwargs"] = dict(kwargs)
        return original(self, **kwargs)

    cls.__call__ = wrapped
    try:
        yield observed
    finally:
        cls.__call__ = original


@contextmanager
def annotate_decoder(module, decoder, manifest):
    import torch

    state = {"step": -1}
    cls = type(decoder)
    original_step = cls._step
    originals = {}

    def step(self, *args, **kwargs):
        state["step"] += 1
        return original_step(self, *args, **kwargs)

    class LabeledKernel:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch_kernel = self.kernel[grid]

            def launch(*args, **kwargs):
                frame = inspect.currentframe().f_back
                # At the final norm/head Python still retains the last layer_idx;
                # use source line ordering to identify the final operations.
                line = frame.f_lineno
                layer = frame.f_locals.get("layer_idx", -1)
                if line >= final_norm_line:
                    layer = -1
                del frame
                label = f"step{state['step']:02d}.layer{layer:02d}.{self.name}.line{line}"
                manifest.append({"index": len(manifest), "operator": label,
                                 "triton_kernel": self.name, "source_line": line,
                                 "grid": str(grid),
                                 "tensors": [{"shape": list(a.shape), "dtype": str(a.dtype)}
                                             for a in args if isinstance(a, torch.Tensor)]})
                torch.cuda.nvtx.range_push(label)
                try:
                    return launch_kernel(*args, **kwargs)
                finally:
                    torch.cuda.nvtx.range_pop()
            return launch

    # Resolve against the checked-out source instead of hard-coded line numbers.
    source, first_line = inspect.getsourcelines(original_step)
    final_norm_line = first_line + next(i for i, line in enumerate(source)
                                        if line.startswith("        _adarms_norm_kernel["))
    qkv_name = "_matmul_gemma_rope_qkv" if hasattr(module, "_matmul_gemma_rope_qkv") and not hasattr(module, "_matmul_rope_qkv") else "_matmul_rope_qkv"
    names = ("_matmul_small_bias", "_adarms_norm_kernel", qkv_name,
             "_attention_prefix_suffix_fused", "_matmul_abt_scale",
             "_softmax_prefix_suffix_mask_vector", "_matmul_small",
             "_matmul_small_res_gate_oproj", "_matmul_small_gate",
             "_matmul_small_res_gate_ffn_down", "_matmul_small_bias_res")
    try:
        cls._step = step
        for name in names:
            originals[name] = getattr(module, name)
            setattr(module, name, LabeledKernel(originals[name], name))
        yield
    finally:
        cls._step = original_step
        for name, original in originals.items():
            setattr(module, name, original)
