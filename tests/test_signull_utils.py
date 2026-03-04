import numpy as np
import pytest
import sys
sys.path.append('.')
from libraries.SigNULL import optimize_waveform_bipolar

def test_optimize_waveform_bipolar_flat():
    # A flat waveform should only keep the first and last points
    times = np.linspace(0, 10, 11)
    amps = np.zeros(11)
    t, a = optimize_waveform_bipolar(times, amps)

    assert len(t) == 2
    assert len(a) == 2
    assert t[0] == 0
    assert t[-1] == 10
    assert a[0] == 0
    assert a[-1] == 0

def test_optimize_waveform_bipolar_peaks():
    # Waveform with a local maximum (peak) and local minimum (valley)
    times = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    amps = np.array([0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5])
    t, a = optimize_waveform_bipolar(times, amps)

    assert t[0] == 0
    assert t[-1] == 7
    # Peak at t=2
    assert 2 in t
    # Valley at t=6
    assert 6 in t

def test_optimize_waveform_bipolar_zero_crossing():
    times = np.array([0, 1, 2, 3, 4])
    amps = np.array([-1, -0.5, 0.5, 1, 1])
    t, a = optimize_waveform_bipolar(times, amps)

    assert t[0] == 0
    assert t[-1] == 4

    # Check zero crossing points
    # At t=1, amp=-0.5. At t=2, amp=0.5. Both should be kept.
    assert 1 in t
    assert 2 in t

def test_optimize_waveform_bipolar_curvature():
    # A curve with acceleration to trigger the curvature fill-in
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    amps = np.array([0.0, 0.1, 0.4, 0.9, 1.6])
    t, a = optimize_waveform_bipolar(times, amps, tol=1e-3)

    assert len(t) > 2 # Should retain points due to curvature

def test_optimize_waveform_bipolar_list_input():
    # The function should convert lists to numpy arrays and not modify the originals
    times_list = [0, 1, 2, 3, 4]
    amps_list = [0, 1, 0, -1, 0]
    t, a = optimize_waveform_bipolar(times_list, amps_list)

    assert isinstance(t, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(t) == 5 # All points are starts, ends, peaks, valleys, or zero crossings
    assert times_list == [0, 1, 2, 3, 4] # Original not modified
    assert amps_list == [0, 1, 0, -1, 0] # Original not modified
