"""
Per-band CSP spatial filters for the HWNet preprocessing pipeline.

CSP (common spatial patterns; Ramoser et al., 2000) is fitted per frequency
band, i.e. filter-bank CSP (Ang et al., 2008). The covariance estimation and
the generalized eigendecomposition are reused from the repository's
`mixnet.preprocessing.FBCSP` (itself adapted from MNE-Python's CSP), but only
the spatial filters are kept: the band-passed signals are *projected* onto
them, so the time axis survives -- no log-variance, no feature selection.

CSP is supervised (it uses `y`), so it must only ever be fitted on the
training fold; `HW_prep.fit_transform()` is the only caller of `fit_csp`.
"""

import numpy as np

from mixnet.preprocessing.FBCSP import FBCSP


def fit_csp(Xf, y, n_csp):
    """Fit one CSP filter bank per band on band-passed TRAINING trials.

    Args:
        Xf (np.ndarray): (n, Nb, C, T) band-passed training trials.
        y (np.ndarray): (n,) labels in {0, 1}.
        n_csp (int): CSP components kept per band (<= C). With two classes
            FBCSP orders them 'alternate', so taking the first n_csp gives
            n_csp / 2 filters from each end of the eigen-spectrum.

    Returns:
        W (np.ndarray): (Nb, n_csp, C) float64 spatial filters.
    """
    n, Nb, C, T = Xf.shape
    if not 0 < n_csp <= C:
        raise ValueError('n_csp must be in [1, {}], got {}'.format(C, n_csp))
    y = np.asarray(y)
    csp = FBCSP(bands=[None] * Nb, smp_freq=None, num_class=len(np.unique(y)),
                n_components=n_csp)
    W = np.empty((Nb, n_csp, C), dtype=np.float64)
    for b in range(Nb):
        # FBCSP's private helpers (name-mangled): trace-normalized per-epoch
        # class covariances -> generalized eigh -> ordered filters.
        covs, weights = csp._FBCSP__calculate_covariance_matrices(
            np.asarray(Xf[:, b], dtype=np.float64), y)
        pick, _ = csp._FBCSP__get_spatial_filter(covs, weights)
        W[b] = pick
    return W


def apply_csp(Xf, W):
    """(n, Nb, C, T) x (Nb, K, C) -> (n, Nb, K, T) float64. Stateless."""
    return np.einsum('bkc,nbct->nbkt', W, np.asarray(Xf, dtype=np.float64))
