import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profiling.runtime import memory_headroom, offline_assets, require_headroom, stream_checkpoint_weights
from profiling.ncu import require_gpu_execution


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="runtime-test-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_offline_assets_require_local_tokenizer(self):
        ckpt, tokenizer = self.root / "checkpoint", self.root / "tokenizer"
        ckpt.mkdir()
        tokenizer.mkdir()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer_config"):
                offline_assets(ckpt, tokenizer)
            (tokenizer / "tokenizer_config.json").write_text("{}")
            offline_assets(ckpt, tokenizer)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            with self.assertRaises(FileNotFoundError):
                offline_assets(ckpt, "remote/repository")

    def test_direct_execution_is_limited_to_authorized_thor(self):
        with patch.dict(os.environ, {}, clear=True):
            for hostname in ("thor0", "fact-thor", "fact-thor.local"):
                with patch("profiling.ncu.socket.gethostname", return_value=hostname):
                    require_gpu_execution()
            with patch("profiling.ncu.socket.gethostname", return_value="deep-space"):
                with self.assertRaisesRegex(RuntimeError, "other hosts require Slurm"):
                    require_gpu_execution()
                with patch.dict(os.environ, {"SLURM_JOB_ID": "test-job"}):
                    require_gpu_execution()

    def test_parent_cgroup_limit_wins_over_host_memory(self):
        proc, cgroup = self.root / "proc", self.root / "cgroup"
        (proc / "self").mkdir(parents=True)
        (proc / "meminfo").write_text("MemAvailable: 104857600 kB\n")
        (proc / "self/cgroup").write_text("0::/job/worker\n")
        (cgroup / "job/worker").mkdir(parents=True)
        for relative, limit, usage in ((".", "max", 0), ("job", 40 * 1024**3, 10 * 1024**3),
                                        ("job/worker", "max", 0)):
            (cgroup / relative / "memory.max").write_text(str(limit))
            (cgroup / relative / "memory.current").write_text(str(usage))
        result = memory_headroom(proc, cgroup)
        self.assertEqual(result["host_available_bytes"], 100 * 1024**3)
        self.assertEqual(result["available_bytes"], 30 * 1024**3)
        with patch("profiling.runtime.memory_headroom", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "Only 30.0 GiB"):
                require_headroom(32)
            self.assertEqual(require_headroom(20), result)
            with self.assertRaises(ValueError):
                require_headroom(float("nan"))

    def test_weight_stream_opens_one_shard_at_a_time(self):
        paths = [self.root / f"model-{i:05d}-of-00002.safetensors" for i in (1, 2)]
        for path in paths:
            path.touch()
        state = {"active": 0, "closed": 0}

        class Reader:
            def __init__(self, path, **kwargs):
                self.path = path
            def __enter__(self):
                state["active"] += 1
                self_outer.assertEqual(state["active"], 1)
                return self
            def __exit__(self, *args):
                state["active"] -= 1
                state["closed"] += 1
            def keys(self):
                return [self.path]
            def get_tensor(self, name):
                return "test tensor"

        self_outer = self
        with patch.dict(sys.modules, {"safetensors": SimpleNamespace(safe_open=Reader)}):
            generator = stream_checkpoint_weights(self.root)
            self.assertEqual(state["active"], 0)
            self.assertEqual(next(generator), (str(paths[0]), "test tensor"))
            self.assertEqual(next(generator), (str(paths[1]), "test tensor"))
            with self.assertRaises(StopIteration):
                next(generator)
        self.assertEqual(state, {"active": 0, "closed": 2})

    def cache_environment(self):
        keys = ("TMPDIR", "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME", "TRITON_CACHE_DIR",
                "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "MPLCONFIGDIR")
        env = {"PATH": os.environ["PATH"]}
        env.update({key: str(self.root / key.lower()) for key in keys})
        return env

    def test_cache_paths_are_preserved_and_downloads_disabled(self):
        env = self.cache_environment()
        result = subprocess.run(["bash", "--noprofile", "--norc", "-c",
                                 'source scripts/cache_env.sh && printf "%s\\n" "$HF_HOME" "$HF_HUB_CACHE" "$HF_HUB_OFFLINE" "$TORCHINDUCTOR_COMPILE_THREADS"'],
                                cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.splitlines(), [env["HF_HOME"], env["HF_HUB_CACHE"], "1", "1"])

    def test_unsafe_cache_fails_before_creating_directories(self):
        env = self.cache_environment()
        env["HF_HOME"] = "/tmp/disallowed-cache"
        result = subprocess.run(["bash", "--noprofile", "--norc", "-c", "source scripts/cache_env.sh"],
                                cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "tmpdir").exists())


if __name__ == "__main__":
    unittest.main()
