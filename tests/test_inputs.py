import unittest

from profiling.inputs import camera_features, validate_camera_mapping


class CameraInputTests(unittest.TestCase):
    def test_regression_short_camera_keys_cannot_silently_pass(self):
        keys = ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb"]
        old = {key.removeprefix("observation.images."): object() for key in keys}
        with self.assertRaisesRegex(ValueError, "silently masked"):
            validate_camera_mapping(old, keys)
        fixed = {key: object() for key in keys}
        self.assertEqual(validate_camera_mapping(fixed, keys)["matched_camera_count"], 3)

    def test_explicit_alias_mapping_and_missing_config_camera(self):
        self.assertTrue(validate_camera_mapping({"front": object()}, ["observation.images.front"],
                        {"front": "observation.images.front"})["all_requested_cameras_present"])
        with self.assertRaisesRegex(ValueError, "only 1"):
            camera_features({"input_features": {"observation.images.front": {}}}, 3)


if __name__ == "__main__":
    unittest.main()
