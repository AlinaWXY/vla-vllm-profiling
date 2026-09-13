import copy
import unittest

from profiling.audit_profile import audit


class ProfileAuditTests(unittest.TestCase):
    def inputs(self):
        contract = {"metrics": ["gpu__time_duration.sum", "lts__t_bytes.sum", "tensor.sum", "add.sum"],
                    "memory": {"level": "l2", "metrics": ["lts__t_bytes.sum"]},
                    "domains": {"bf16_tensor": {"terms": [{"metric": "tensor.sum", "weight": 1}]},
                                "fp32_simt": {"terms": [{"metric": "add.sum", "weight": 1}]}}}
        kernel = {"id": "0", "kernel": "_matmul", "operator": "step00.layer00._matmul.line10",
                  "metrics": {"gpu__time_duration.sum": (1000, "ns"),
                              "lts__t_bytes.sum": (200, "byte"), "tensor.sum": (800, "op"),
                              "add.sum": (0, "inst")}}
        benchmark = {"profiled": True, "workload": {"steps": 1},
                     "operator_manifest": [{"operator": kernel["operator"]}]}
        return [kernel], benchmark, contract

    def test_exact_coverage_and_no_precision_double_count(self):
        kernels, benchmark, contract = self.inputs()
        report, hotspots = audit(kernels, benchmark, contract)
        self.assertEqual(report["ncu_kernel_duration_sum_ms"], .001)
        self.assertEqual(report["memory_bytes_sum"], 200)
        self.assertEqual(report["counted_flops_by_domain"], {"bf16_tensor": 800, "fp32_simt": 0})
        self.assertEqual(hotspots[0]["ncu_time_share_pct"], 100)

    def test_missing_duplicate_or_wrong_kernel_is_rejected(self):
        kernels, benchmark, contract = self.inputs()
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            audit(kernels * 2, benchmark, contract)
        bad = copy.deepcopy(kernels)
        bad[0]["kernel"] = "another_kernel"
        with self.assertRaisesRegex(ValueError, "name does not match"):
            audit(bad, benchmark, contract)
        del kernels[0]["metrics"]["tensor.sum"]
        with self.assertRaisesRegex(ValueError, "Missing/invalid metric"):
            audit(kernels, benchmark, contract)


if __name__ == "__main__":
    unittest.main()
