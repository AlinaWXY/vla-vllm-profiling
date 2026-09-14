"""Validate observation camera keys against the upstream preprocessing contract."""
from __future__ import annotations


def camera_features(config, cameras):
    keys = [key for key in config["input_features"] if key.startswith("observation.images.")]
    if len(keys) < cameras:
        raise ValueError(f"Requested {cameras} cameras but checkpoint declares only {len(keys)}")
    return keys[:cameras]


def validate_camera_mapping(images, feature_keys, key_map=None):
    key_map = key_map or {}
    canonical = {key_map.get(key, key): image for key, image in images.items()}
    missing = [key for key in feature_keys if key not in canonical or canonical[key] is None]
    if missing:
        raise ValueError(f"Images would be silently masked by upstream preprocessing: {missing}")
    return {"provided_keys": list(images), "required_keys": list(feature_keys),
            "matched_camera_count": len(feature_keys), "all_requested_cameras_present": True}
