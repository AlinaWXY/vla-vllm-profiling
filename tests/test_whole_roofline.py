import unittest

from profiling.whole_roofline import combine


class WholeRangeTests(unittest.TestCase):
    def fixture(self):
        summary = {"totals": {
            "vlm": {"flops": {"bf16_tensor": 6000, "fp32_simt": 200}, "kernel_time_sum_ms": 99},
            "action_expert": {"flops": {"bf16_tensor": 4000, "fp32_simt": 300}, "kernel_time_sum_ms": 88}}}
        ranges = [{"id": str(i), "kernel": name, "metrics": {
            "gpu__time_duration.sum": (time, "usecond"), "lts__t_bytes.sum": (traffic, "byte"),
            "sm__ops_path_tensor_src_bf16_dst_fp32.sum": (flops, "op")}}
            for i, (name, time, traffic, flops) in enumerate([
                ("vlm", 2, 1000, 6000), ("action_expert", 3, 2000, 4000), ("vla", 4, 2500, 10000)])]
        return ranges, summary

    def test_combined_scope_uses_measured_range_time_and_traffic(self):
        ranges, summary = self.fixture()
        rows = combine(ranges, summary, ["vlm", "action_expert", "vla"])
        total, scalar = rows[-2:]
        self.assertEqual(total["ai_flops_per_byte"], 4)
        self.assertAlmostEqual(total["performance_tflops"], .0025)
        self.assertEqual(total["range_duration_ms"], .004)
        self.assertEqual(scalar["flops"], 500)
        self.assertIn("kernel-replay", scalar["flops_source"])

    def test_refuses_to_join_different_workloads(self):
        ranges, summary = self.fixture()
        ranges[-1]["metrics"]["sm__ops_path_tensor_src_bf16_dst_fp32.sum"] = (9999, "op")
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            combine(ranges, summary, ["vlm", "action_expert", "vla"])

    def test_missing_range_counter_is_not_zero(self):
        ranges, summary = self.fixture()
        del ranges[-1]["metrics"]["lts__t_bytes.sum"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            combine(ranges, summary, ["vlm", "action_expert", "vla"])


if __name__ == "__main__":
    unittest.main()
