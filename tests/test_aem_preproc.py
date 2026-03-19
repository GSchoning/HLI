import numpy as np
import pytest
from libraries.AEM_preproc import calculate_vertical_correction_numpy

def test_vertical_correction_scalars():
    # Regular cases
    assert calculate_vertical_correction_numpy(0, 10) == pytest.approx(0.0)
    assert calculate_vertical_correction_numpy(45, 10) == pytest.approx(10.0)
    assert calculate_vertical_correction_numpy(-45, 10) == pytest.approx(-10.0)

    # Edge cases (+/- 90) should return array(None, dtype=object) or None
    assert calculate_vertical_correction_numpy(90, 10) == None
    assert calculate_vertical_correction_numpy(-90, 10) == None

    # Clipped edge cases
    assert calculate_vertical_correction_numpy(100, 10) == None
    assert calculate_vertical_correction_numpy(-100, 10) == None


def test_vertical_correction_arrays():
    # Array vs Scalar
    angles = np.array([0, 45, -45, 90, -90, 100])
    dist = 10

    expected = [0.0, 10.0, -10.0, None, None, None]
    result = calculate_vertical_correction_numpy(angles, dist)

    for r, e in zip(result, expected):
        if e is None:
            assert r is None
        else:
            assert r == pytest.approx(e)

    # Array vs Array
    dist_arr = np.array([10, 20, 10, 10, 10, 10])
    expected_arr = [0.0, 20.0, -10.0, None, None, None]
    result_arr = calculate_vertical_correction_numpy(angles, dist_arr)

    for r, e in zip(result_arr, expected_arr):
        if e is None:
            assert r is None
        else:
            assert r == pytest.approx(e)


def test_vertical_correction_invalid_inputs():
    # Non-numeric / non-ndarray inputs
    assert calculate_vertical_correction_numpy("45", 10) is None
    assert calculate_vertical_correction_numpy(45, "10") is None
    assert calculate_vertical_correction_numpy([45], 10) is None
    assert calculate_vertical_correction_numpy(45, [10]) is None

    # Mismatched shapes
    assert calculate_vertical_correction_numpy(np.array([45, 0]), np.array([10])) is None
    assert calculate_vertical_correction_numpy(np.array([45, 0]), np.array([10, 20, 30])) is None
