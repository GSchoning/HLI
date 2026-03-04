import numpy as np
import pytest
import scipy.sparse as sp
from libraries.SigNULL import get_cholesky_decomposition

class MockMesh:
    def __init__(self, nC):
        self.nC = nC
        self.n_cells = nC

def test_get_cholesky_decomposition_single_cell():
    mesh = MockMesh(1)
    L_mat = get_cholesky_decomposition(mesh, 2.0)
    assert sp.issparse(L_mat)
    assert L_mat.shape == (1, 1)
    np.testing.assert_allclose(L_mat.toarray(), np.eye(1))

def test_get_cholesky_decomposition_shape():
    mesh = MockMesh(5)
    L_mat = get_cholesky_decomposition(mesh, 2.0)
    assert L_mat.shape == (5, 5)

def test_get_cholesky_decomposition_reconstruction():
    nC = 5
    corr_factor = 2.0
    mesh = MockMesh(nC)
    L_mat = get_cholesky_decomposition(mesh, corr_factor)

    idx = np.arange(nC)
    index_dist = np.abs(idx[:, None] - idx[None, :])
    expected_C = np.exp(-1.0 * (index_dist / corr_factor)) + np.eye(nC) * 1e-6

    reconstructed_C = L_mat @ L_mat.T
    np.testing.assert_allclose(reconstructed_C, expected_C, rtol=1e-5, atol=1e-5)

def test_get_cholesky_decomposition_min_corr_factor():
    mesh = MockMesh(5)
    L_mat1 = get_cholesky_decomposition(mesh, 0.05)
    L_mat2 = get_cholesky_decomposition(mesh, 0.1)

    np.testing.assert_allclose(L_mat1, L_mat2)

def test_get_cholesky_decomposition_svd_fallback(monkeypatch):
    def mock_cholesky(*args, **kwargs):
        raise np.linalg.LinAlgError("Mock error")

    monkeypatch.setattr(np.linalg, "cholesky", mock_cholesky)

    mesh = MockMesh(5)
    L_mat = get_cholesky_decomposition(mesh, 2.0)

    assert L_mat.shape == (5, 5)
