import numpy as np
import pytest
from libraries.HIES import adjust_dtype

def test_adjust_dtype_np_integer():
    assert adjust_dtype(np.int64(42)) == 42
    assert isinstance(adjust_dtype(np.int64(42)), int)
    assert adjust_dtype(np.int32(10)) == 10
    assert isinstance(adjust_dtype(np.int32(10)), int)

def test_adjust_dtype_np_floating():
    assert adjust_dtype(np.float64(3.14)) == 3.14
    assert isinstance(adjust_dtype(np.float64(3.14)), float)
    assert adjust_dtype(np.float32(2.5)) == pytest.approx(2.5)
    assert isinstance(adjust_dtype(np.float32(2.5)), float)

def test_adjust_dtype_np_ndarray():
    arr = np.array([1, 2, 3])
    result = adjust_dtype(arr)
    assert result == [1, 2, 3]
    assert isinstance(result, list)

    arr_2d = np.array([[1, 2], [3, 4]])
    result_2d = adjust_dtype(arr_2d)
    assert result_2d == [[1, 2], [3, 4]]
    assert isinstance(result_2d, list)

def test_adjust_dtype_python_types():
    assert adjust_dtype(42) == 42
    assert adjust_dtype(3.14) == 3.14
    assert adjust_dtype("hello") == "hello"
    assert adjust_dtype([1, 2, 3]) == [1, 2, 3]
    assert adjust_dtype(None) is None

def test_adjust_dtype_edge_cases():
    # Empty array
    arr = np.array([])
    assert adjust_dtype(arr) == []

    # NaN and Inf
    assert np.isnan(adjust_dtype(np.float64(np.nan)))
    assert np.isinf(adjust_dtype(np.float64(np.inf)))
