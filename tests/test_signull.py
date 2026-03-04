import numpy as np
import pytest
from libraries.SigNULL import cdf_for_value

def test_cdf_for_value_normal():
    data = [1, 2, 3, 4, 5]
    assert cdf_for_value(data, 3) == 0.6  # 3 elements <= 3: 3/5 = 0.6
    assert cdf_for_value(data, 2.5) == 0.4 # 2 elements <= 2.5: 2/5 = 0.4

def test_cdf_for_value_all_less_equal():
    data = [1, 2, 3]
    assert cdf_for_value(data, 5) == 1.0  # All elements <= 5

def test_cdf_for_value_all_greater():
    data = [5, 6, 7]
    assert cdf_for_value(data, 3) == 0.0  # No elements <= 3

def test_cdf_for_value_empty():
    data = []
    assert cdf_for_value(data, 5) == 0.0  # Empty array should return 0.0

def test_cdf_for_value_numpy_array():
    data = np.array([10, 20, 30, 40])
    assert cdf_for_value(data, 25) == 0.5  # 2 elements <= 25: 2/4 = 0.5

def test_cdf_for_value_negative_numbers():
    data = [-5, -2, 0, 2, 5]
    assert cdf_for_value(data, -1) == 0.4  # -5, -2 are <= -1: 2/5 = 0.4
    assert cdf_for_value(data, -5) == 0.2  # -5 is <= -5: 1/5 = 0.2

def test_cdf_for_value_identical_elements():
    data = [2, 2, 2, 2]
    assert cdf_for_value(data, 2) == 1.0
    assert cdf_for_value(data, 1) == 0.0

def test_cdf_for_value_large_array():
    data = np.arange(1, 101)  # 1 to 100
    assert cdf_for_value(data, 50) == 0.5
    assert cdf_for_value(data, 10) == 0.1

def test_cdf_for_value_floats():
    data = [1.1, 2.2, 3.3, 4.4]
    assert cdf_for_value(data, 3.0) == 0.5
    assert cdf_for_value(data, 4.4) == 1.0
