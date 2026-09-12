import csv
import io
from pathlib import Path
import tempfile
import unittest

from profiling.ncu import FP32_METRICS, select_contract
from profiling.roofline import analyze, attach_operators, load_ncu


CONTRACT = {"domains": {"bf16_tensor": {"terms": [{"metric": "tensor.sum", "weight": 1}]},
                        "fp32_simt": {"terms": [{"metric": "fma.sum", "weight": 2}]}}}


class RooflineTests(unittest.TestCase):
    def kernel(self, **changes):
        k = {"id": "0", "kernel": "test", "metrics": {
            "gpu__time_duration.sum": (2, "usecond"),
            "dram__bytes_read.sum": (600, "byte"),
            "dram__bytes_write.sum": (400, "byte"),
            "tensor.sum": (4000, "op"), "fma.sum": (500, "inst")}}
        k["metrics"].update(changes)
        return k

    def test_units_precision_and_fma(self):
        bf16, fp32 = analyze([self.kernel()], CONTRACT)
        self.assertEqual(bf16["duration_ns"], 2000)
        self.assertEqual(bf16["ai_flops_per_byte"], 4)
        self.assertAlmostEqual(bf16["performance_tflops"], .002)
        self.assertEqual(fp32["flops"], 1000)
        self.assertEqual(fp32["ai_flops_per_byte"], 1)

    def test_missing_is_not_zero(self):
        k = self.kernel()
        del k["metrics"]["tensor.sum"]
        row = analyze([k], CONTRACT)[0]
        self.assertIsNone(row["flops"])
        self.assertIsNone(row["performance_tflops"])
        self.assertIn("missing_flop_metrics", row["status"])

    def test_zero_traffic_is_retained_but_not_plotted(self):
        k = self.kernel(**{"dram__bytes.sum": (0, "byte")})
        row = analyze([k], CONTRACT)[0]
        self.assertIsNone(row["ai_flops_per_byte"])
        self.assertEqual(row["dram_bytes"], 0)

    def test_unknown_units_fail(self):
        with self.assertRaisesRegex(ValueError, "Unsupported time unit"):
            analyze([self.kernel(**{"gpu__time_duration.sum": (2, "cycles")})], CONTRACT)

    def test_csv_kernel_identity_and_unavailable_counter(self):
        stream = io.StringIO()
        stream.write("==PROF== Connected to process 10\n")
        w = csv.writer(stream)
        w.writerow(["ID", "Process ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
        w.writerow([0, 10, "gemm, templated", "tensor.sum", "op", "1,234"])
        w.writerow([0, 11, "gemm", "tensor.sum", "op", "n/a"])
        # Use the project directory, never the system /tmp or a home directory.
        with tempfile.TemporaryDirectory(prefix="pi05-tests-", dir=Path(__file__).parent) as td:
            path = Path(td) / "input.csv"
            path.write_text(stream.getvalue())
            kernels = load_ncu(path)
        self.assertEqual(len(kernels), 2)
        self.assertEqual(kernels[0]["metrics"]["tensor.sum"], (1234, "op"))
        self.assertIsNone(kernels[1]["metrics"]["tensor.sum"][0])

    def test_duplicate_conflicts_fail(self):
        with tempfile.TemporaryDirectory(prefix="pi05-tests-", dir=Path(__file__).parent) as td:
            path = Path(td) / "input.csv"
            path.write_text('"ID","Metric Name","Metric Value"\n"0","a","1"\n"0","a","2"\n')
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                load_ncu(path)

    def test_actual_thor_ncu_wide_export(self):
        path = Path(__file__).parents[1] / "results/processed/environment_sm110a/ncu_probe.csv"
        kernels = load_ncu(path)
        self.assertEqual(len(kernels), 1)
        self.assertEqual(kernels[0]["kernel"], "write_indices")
        self.assertEqual(kernels[0]["metrics"]["gpu__time_duration.sum"], (2496, "ns"))
        self.assertEqual(kernels[0]["metrics"]["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"],
                         (1024, "inst"))

    def test_no_double_count_tensor_parent_and_sparse_child(self):
        query = "\n".join(["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
                            *FP32_METRICS, "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
                            "sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum"])
        contract = select_contract(query)
        self.assertEqual(len(contract["domains"]["bf16_tensor"]["terms"]), 1)
        self.assertEqual(contract["domains"]["bf16_tensor"]["terms"][0]["weight"], 1)

    def test_unknown_architecture_requires_review(self):
        with self.assertRaisesRegex(ValueError, "manual review"):
            select_contract("gpu__time_duration.sum")

    def test_l2_is_explicit_and_never_labeled_dram(self):
        query = "\n".join(["gpu__time_duration.sum", "lts__t_bytes.sum", *FP32_METRICS,
                           "sm__ops_path_tensor_src_bf16_dst_fp32.sum"])
        with self.assertRaisesRegex(ValueError, "manual review"):
            select_contract(query)
        contract = select_contract(query, "l2")
        kernel = self.kernel(**{"lts__t_bytes.sum": (2000, "byte"),
                              "sm__ops_path_tensor_src_bf16_dst_fp32.sum": (4000, "op")})
        row = analyze([kernel], contract)[0]
        self.assertEqual(row["memory_level"], "l2")
        self.assertEqual(row["ai_flops_per_byte"], 2)
        self.assertIsNone(row["dram_bytes"])
        contract["memory"]["level"] = "dram"
        with self.assertRaisesRegex(ValueError, "must not be labeled DRAM"):
            analyze([kernel], contract)

    def test_operator_join_rejects_different_reports(self):
        kernel = self.kernel()
        annotated = self.kernel()
        annotated["kernel"] = "step00.layer00.qkv"
        attach_operators([kernel], [annotated])
        self.assertEqual(kernel["operator"], "step00.layer00.qkv")
        annotated["metrics"]["tensor.sum"] = (123, "op")
        with self.assertRaisesRegex(ValueError, "metrics differ"):
            attach_operators([kernel], [annotated])

    def test_invalid_contracts_do_not_produce_false_points(self):
        for terms in ([], [{"metric": "tensor.sum.per_second", "weight": 1}],
                      [{"metric": "tensor.sum", "weight": 0}],
                      [{"metric": "tensor.sum", "weight": 1}] * 2):
            with self.assertRaises(ValueError):
                analyze([self.kernel()], {"domains": {"bf16_tensor": {"terms": terms}}})


if __name__ == "__main__":
    unittest.main()
