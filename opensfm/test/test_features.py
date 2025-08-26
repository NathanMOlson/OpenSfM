# pyre-strict
import numpy as np
import pytest
from opensfm import config, features


def test_16bit_hahog() -> None:
    conf = config.default_config()
    conf["feature_type"] = "HAHOG"
    conf["hahog_peak_threshold"] = 0.00001
    conf["hahog_edge_threshold"] = 10
    conf["hahog_normalize_to_uchar"] = True

    assert features.does_type_support_any_depth(conf["feature_type"])

    np.random.seed(1)
    img = (np.random.rand(64, 64) * 65535).astype(np.uint16)

    points, f, c = features.extract_features(
        image=img, config=conf, is_panorama=False)

    assert points.shape == (37, 4)


def test_8bit_hahog() -> None:
    conf = config.default_config()
    conf["feature_type"] = "HAHOG"
    conf["hahog_peak_threshold"] = 0.00001
    conf["hahog_edge_threshold"] = 10
    conf["hahog_normalize_to_uchar"] = True

    np.random.seed(1)
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)

    points, f, c = features.extract_features(
        image=img, config=conf, is_panorama=False)

    assert points.shape == (37, 4)


def test_16bit_dspsift() -> None:
    conf = config.default_config()
    conf["feature_type"] = "DSPSIFT"
    conf["sift_peak_threshold"] = 0.001
    conf["sift_edge_threshold"] = 10
    conf["sift_nfeatures"] = 0
    conf["sift_octave_layers"] = 3
    conf["sift_sigma"] = 1.6

    assert features.does_type_support_any_depth(conf["feature_type"])

    np.random.seed(1)
    img = (np.random.rand(64, 64) * 65535).astype(np.uint16)

    points, f, c = features.extract_features(
        image=img, config=conf, is_panorama=False)

    assert points.shape == (5, 4)


def test_8bit_dspsfit() -> None:
    conf = config.default_config()
    conf["feature_type"] = "DSPSIFT"
    conf["sift_peak_threshold"] = 0.001
    conf["sift_edge_threshold"] = 10
    conf["sift_nfeatures"] = 0
    conf["sift_octave_layers"] = 3
    conf["sift_sigma"] = 1.6

    np.random.seed(1)
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)

    points, f, c = features.extract_features(
        image=img, config=conf, is_panorama=False)

    assert points.shape == (5, 4)


def test_16bit_sift() -> None:
    conf = config.default_config()
    conf["feature_type"] = "SIFT"
    conf["sift_peak_threshold"] = 0.1
    conf["sift_edge_threshold"] = 10
    conf["sift_nfeatures"] = 0
    conf["sift_octave_layers"] = 3
    conf["sift_sigma"] = 1.6

    assert features.does_type_support_any_depth(conf["feature_type"]) == False

    np.random.seed(1)
    img = (np.random.rand(64, 64) * 65535).astype(np.uint16)

    with pytest.raises(AssertionError):
        points, f, c = features.extract_features(
            image=img, config=conf, is_panorama=False)


def test_8bit_sift() -> None:
    conf = config.default_config()
    conf["feature_type"] = "SIFT"
    conf["sift_peak_threshold"] = 0.1
    conf["sift_edge_threshold"] = 10
    conf["sift_nfeatures"] = 0
    conf["sift_octave_layers"] = 3
    conf["sift_sigma"] = 1.6

    np.random.seed(1)
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)

    points, f, c = features.extract_features(
        image=img, config=conf, is_panorama=False)

    assert points.shape == (9, 4)
