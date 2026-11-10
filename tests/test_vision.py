import numpy as np
import pytest
from vision.corruptions import CORRUPTIONS, apply_corruption
from vision.calibration import apply_temperature, ece, fit_temperature, iou, match_predictions
from vision.temporal import ConfidenceSmoother, flicker_rate


def img():
    return (np.random.default_rng(1).random((120, 160, 3)) * 200 + 20).astype(np.uint8)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_corruption_shape_dtype_and_severity_monotonic(name):
    a = img()
    assert np.array_equal(apply_corruption(a, name, 0), a)
    d = [np.abs(apply_corruption(a, name, s).astype(int) - a.astype(int)).mean() for s in (1, 3, 5)]
    assert d[0] < d[1] < d[2]
    assert apply_corruption(a, name, 3).shape == a.shape


def test_corruption_validation():
    with pytest.raises(ValueError):
        apply_corruption(img(), "nope", 1)
    with pytest.raises(ValueError):
        apply_corruption(img(), "blur", 9)
    with pytest.raises(ValueError):
        apply_corruption(np.zeros((4, 4), np.uint8), "blur", 1)


def test_noise_is_seeded():
    a = img()
    assert np.array_equal(apply_corruption(a, "noise", 3, seed=5), apply_corruption(a, "noise", 3, seed=5))


def test_temperature_identity_and_direction():
    p = np.array([0.6, 0.9])
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-6)
    assert (apply_temperature(p, 2.0) < p).all()   # softer


def test_ece_perfect_and_fit_temperature_reduces_overconfidence():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.5, 0.9, 4000)
    correct = (rng.random(4000) < true_p).astype(int)
    overconf = apply_temperature(true_p, 0.4)           # sharpen -> over-confident
    e_before, _ = ece(overconf, correct)
    T = fit_temperature(overconf, correct)
    e_after, _ = ece(apply_temperature(overconf, T), correct)
    assert T > 1.0 and e_after < e_before
    assert ece(np.array([1.0, 1.0]), np.array([1, 1]))[0] == 0.0


def test_fit_temperature_needs_both_classes():
    with pytest.raises(ValueError):
        fit_temperature(np.linspace(.1, .9, 20), np.ones(20))


def test_iou_and_matching():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
    c, ok, missed = match_predictions([(0, .9, (0, 0, 10, 10)), (0, .8, (0, 0, 10, 10)), (1, .7, (50, 50, 60, 60))], [(0, (0, 0, 10, 10))])
    assert ok.tolist() == [1, 0, 0] and missed == 0   # duplicate + wrong class are false positives


def test_smoother_and_flicker():
    s = ConfidenceSmoother(0.5)
    assert s.update(1, 1.0) == 1.0 and s.update(1, 0.0) == 0.5
    assert flicker_rate([True, True, True]) == 0 and flicker_rate([True, False, True]) == 1.0
