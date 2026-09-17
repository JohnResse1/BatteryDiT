"""
Diagnostic feature extraction from battery voltage curves.

All functions operate on numpy arrays of shape (batch, seq_len) or (seq_len,).
Voltage values are assumed to be in physical units (Volts), already denormalized.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def _to_2d(V: np.ndarray) -> np.ndarray:
    """Ensure input is (batch, seq_len)."""
    V = np.asarray(V, dtype=np.float64)
    if V.ndim == 1:
        V = V[np.newaxis, :]
    if V.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {V.shape}")
    return V


def _interpolate_threshold_index(V: np.ndarray, threshold: float) -> np.ndarray:
    """
    For each sample, find the fractional time index where V crosses `threshold`.
    V is assumed to be monotonically increasing along axis=1.

    Returns: (batch,) array of fractional indices in [0, seq_len-1].
    """
    B, L = V.shape
    result = np.empty(B, dtype=np.float64)

    for i in range(B):
        v = V[i]
        # Find first index where V >= threshold
        mask = v >= threshold
        idx = np.argmax(mask)

        if not mask[idx]:
            # Never reaches threshold — return seq_len
            result[i] = float(L - 1)
        elif idx == 0:
            result[i] = 0.0
        else:
            # Linear interpolation between idx-1 and idx
            v_lo, v_hi = v[idx - 1], v[idx]
            if v_hi - v_lo < 1e-12:
                result[i] = float(idx)
            else:
                frac = (threshold - v_lo) / (v_hi - v_lo)
                result[i] = float(idx - 1) + frac

    return result


def Q_at_V(V: np.ndarray, voltages: Optional[List[float]] = None) -> np.ndarray:
    """
    Fixed-voltage capacity: normalized time index at each voltage threshold.

    For constant-current charging, time index is proportional to capacity (Q ≈ I·t),
    so the normalized time index at a given voltage approximates Q(V).

    Args:
        V: (batch, seq_len) or (seq_len,) voltage curve(s), monotonically increasing.
        voltages: list of voltage thresholds. Default: 6 evenly spaced points
                  spanning the 5th-to-95th percentile range of the data.

    Returns:
        (batch, len(voltages)) array of normalized time indices in [0, 1].
    """
    V = _to_2d(V)
    if voltages is None:
        # Auto-range: use 5 evenly spaced points excluding extremes
        v_min, v_max = np.percentile(V, [10, 90])
        voltages = list(np.linspace(v_min, v_max, 5))

    L = V.shape[1]
    result = np.zeros((V.shape[0], len(voltages)), dtype=np.float64)
    for j, thr in enumerate(voltages):
        idx = _interpolate_threshold_index(V, thr)
        result[:, j] = idx / (L - 1)  # normalize to [0, 1]

    return result


def delta_Q(V: np.ndarray, V_low: Optional[float] = None,
            V_high: Optional[float] = None) -> np.ndarray:
    """
    Voltage-window capacity: trapezoidal integral of V between two voltage bounds,
    normalized by the total area under the curve.

    Args:
        V: (batch, seq_len) or (seq_len,) voltage curve.
        V_low, V_high: voltage bounds. Default: 10th and 90th percentiles.

    Returns:
        (batch, 1) array of normalized window capacity ratios in [0, 1].
    """
    V = _to_2d(V)
    B, L = V.shape

    # Total area under curve (trapezoidal)
    total_area = np.trapz(V, axis=1)  # (B,)

    # Find indices bounding the voltage window
    if V_low is None:
        V_low = np.percentile(V, 10)
    if V_high is None:
        V_high = np.percentile(V, 90)

    i_low = _interpolate_threshold_index(V, V_low)   # (B,)
    i_high = _interpolate_threshold_index(V, V_high)  # (B,)

    # Clip to valid range
    i_low = np.clip(i_low, 0, L - 1)
    i_high = np.clip(i_high, 0, L - 1)

    window_area = np.zeros(B, dtype=np.float64)
    for b in range(B):
        lo = int(np.floor(i_low[b]))
        hi = int(np.ceil(i_high[b]))
        lo = max(0, min(lo, L - 2))
        hi = max(lo + 1, min(hi, L - 1))
        window_area[b] = np.trapz(V[b, lo:hi + 1])

    ratio = np.where(total_area > 1e-12, window_area / total_area, 0.0)
    return ratio[:, np.newaxis]


def curve_area(V: np.ndarray) -> np.ndarray:
    """
    Total area under the voltage charging curve, normalized by
    seq_len * (v_max - v_min).

    Args:
        V: (batch, seq_len) or (seq_len,).

    Returns:
        (batch, 1) normalized area.
    """
    V = _to_2d(V)
    total = np.trapz(V, axis=1)  # (B,)
    L = V.shape[1]
    v_range = V.max(axis=1) - V.min(axis=1)
    denom = (L - 1) * v_range
    norm = np.where(denom > 1e-12, total / denom, 0.0)
    return norm[:, np.newaxis]


def plateau_slope(V: np.ndarray, v_min: Optional[float] = None,
                  v_max: Optional[float] = None) -> np.ndarray:
    """
    Slope of the voltage plateau region, computed by linear fit
    over the voltage segment between v_min and v_max.

    Args:
        V: (batch, seq_len) or (seq_len,).
        v_min, v_max: voltage bounds for plateau region.
                      Default: 30th and 70th percentiles.

    Returns:
        (batch, 1) slope values (dV / dt).
    """
    V = _to_2d(V)
    B, L = V.shape

    if v_min is None:
        v_min = float(np.percentile(V, 30))
    if v_max is None:
        v_max = float(np.percentile(V, 70))

    slopes = np.empty(B, dtype=np.float64)
    t = np.arange(L, dtype=np.float64)

    for b in range(B):
        v = V[b]
        # Indices in plateau region
        mask = (v >= v_min) & (v <= v_max)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            slopes[b] = 0.0
        else:
            # Linear fit: V = a * t + b  ->  a is slope
            A = np.vstack([t[idx], np.ones_like(idx)]).T
            coeffs, *_ = np.linalg.lstsq(A, v[idx], rcond=None)
            slopes[b] = coeffs[0]

    return slopes[:, np.newaxis]


def ic_curve(V: np.ndarray, dV: float = 0.01,
             smooth_window: int = 11) -> np.ndarray:
    """
    Incremental Capacity curve: dQ/dV ≈ 1 / (dV/dt).

    For constant-current charging, dQ = I·dt, so dQ/dV = I / (dV/dt).
    Since I is roughly constant, the shape of dQ/dV is proportional to 1/(dV/dt).

    Args:
        V: (batch, seq_len) or (seq_len,).
        dV: small voltage offset for numerical derivative (used for clamping).
        smooth_window: moving average window size (odd number).

    Returns:
        (batch, seq_len - 1) IC curve.
    """
    V = _to_2d(V)
    B, L = V.shape

    # dV/dt along time axis
    dv = np.diff(V, axis=1)  # (B, L-1)

    # dQ/dV ≈ I / (dV/dt); since I is constant, use 1/(dV/dt)
    eps = dV / 10.0
    ic = np.where(np.abs(dv) > eps, 1.0 / np.abs(dv), 0.0)

    # Clamp outliers to 99th percentile
    for b in range(B):
        p99 = np.percentile(ic[b], 99)
        if p99 > 0:
            ic[b] = np.clip(ic[b], 0, p99 * 3)

    # Smooth with moving average
    if smooth_window > 1 and smooth_window < L - 1:
        window = np.ones(smooth_window) / smooth_window
        ic = np.apply_along_axis(
            lambda x: np.convolve(x, window, mode='same'), axis=1, arr=ic
        )

    return ic.astype(np.float64)


def ic_peaks(ic: np.ndarray, V_mid: Optional[np.ndarray] = None,
             n_peaks: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract IC peak positions and areas.

    Uses simple local-maximum peak detection (no scipy dependency).

    Args:
        ic: (batch, seq_len-1) IC curve from ic_curve().
        V_mid: (batch, seq_len-1) midpoint voltages.
               If None, uses axis-1 index as proxy.
        n_peaks: number of top peaks to extract.

    Returns:
        peak_positions: (batch, n_peaks) voltage at each peak (NaN if peak not found).
        peak_areas: (batch, n_peaks) area under each peak (NaN if not found).
    """
    B, Lm1 = ic.shape

    if V_mid is None:
        V_mid = np.arange(Lm1, dtype=np.float64)[np.newaxis, :].repeat(B, axis=0)

    positions = np.full((B, n_peaks), np.nan, dtype=np.float64)
    areas = np.full((B, n_peaks), np.nan, dtype=np.float64)

    for b in range(B):
        # Find local maxima
        signal = ic[b]
        # Pad to detect peaks at boundaries
        padded = np.concatenate([[0], signal, [0]])
        is_peak = (padded[1:-1] > padded[:-2]) & (padded[1:-1] >= padded[2:])

        peak_indices = np.where(is_peak)[0]
        if len(peak_indices) == 0:
            continue

        peak_heights = signal[peak_indices]
        # Sort by height descending
        order = np.argsort(peak_heights)[::-1]

        for j in range(min(n_peaks, len(order))):
            pi = peak_indices[order[j]]
            positions[b, j] = V_mid[b, pi]

            # Area: simple triangular approximation around the peak
            left = max(0, pi - 5)
            right = min(Lm1, pi + 5)
            areas[b, j] = np.trapz(signal[left:right])

    return positions, areas


def extract_all_features(V: np.ndarray, V_clean: Optional[np.ndarray] = None,
                         feature_set: str = 'basic',
                         voltages: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    """
    Extract all diagnostic features from voltage curve(s).

    Args:
        V: (batch, seq_len) or (seq_len,) voltage curve(s).
        V_clean: clean reference for threshold-based features.
                 If None, uses V itself.
        feature_set: 'basic' (Q_at_V, delta_Q, curve_area, plateau_slope)
                   | 'ic' (ic_peaks only)
                   | 'all' (all features).
        voltages: custom voltage thresholds for Q_at_V.

    Returns:
        dict with keys (subset depending on feature_set):
            'Q_at_V': (batch, n_thresholds)
            'delta_Q': (batch, 1)
            'curve_area': (batch, 1)
            'plateau_slope': (batch, 1)
            'ic_peak_pos': (batch, n_peaks)
            'ic_peak_area': (batch, n_peaks)
    """
    V = _to_2d(V)
    ref = V_clean if V_clean is not None else V

    features = {}

    if feature_set in ('basic', 'all'):
        features['Q_at_V'] = Q_at_V(V, voltages=voltages)
        features['delta_Q'] = delta_Q(V)
        features['curve_area'] = curve_area(V)
        features['plateau_slope'] = plateau_slope(V)

    if feature_set in ('ic', 'all'):
        ic = ic_curve(V)
        V_mid = (V[:, :-1] + V[:, 1:]) / 2.0
        pos, area = ic_peaks(ic, V_mid, n_peaks=2)
        features['ic_peak_pos'] = pos
        features['ic_peak_area'] = area

    return features


def compute_frr(f_clean: np.ndarray, f_corrupt: np.ndarray,
                f_recon: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute Feature Recovery Ratio.

    FRR = (E_corrupt - E_recon) / (E_corrupt + eps)

    where E_corrupt = |f_corrupt - f_clean|
          E_recon   = |f_recon - f_clean|

    FRR > 0: recovery helps.
    FRR = 1: perfect recovery (E_recon = 0).
    FRR = 0: no improvement over corrupted.
    FRR < 0: reconstruction made it worse.

    Handles NaN features by returning NaN in those dimensions.

    Args:
        f_clean, f_corrupt, f_recon: each (batch, D) or (batch,) or scalar.
            All must have the same shape.
        eps: small constant to avoid division by zero.

    Returns:
        Same shape as inputs — per-sample, per-dimension FRR.
    """
    f_clean = np.asarray(f_clean, dtype=np.float64)
    f_corrupt = np.asarray(f_corrupt, dtype=np.float64)
    f_recon = np.asarray(f_recon, dtype=np.float64)

    E_corrupt = np.abs(f_corrupt - f_clean)
    E_recon = np.abs(f_recon - f_clean)

    # Where E_corrupt is near zero (feature already accurate in corrupted data),
    # FRR is undefined; set to 0 (no improvement needed).
    denom = E_corrupt + eps
    frr = (E_corrupt - E_recon) / denom
    frr = np.where(E_corrupt < eps, 0.0, frr)

    return frr


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (N, N) from (N, L) array."""
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    dist_sq = sq + sq.T - 2.0 * X @ X.T
    np.clip(dist_sq, 0, None, out=dist_sq)
    return np.sqrt(dist_sq)


def _kendall_tau_approx(D_clean: np.ndarray, D_recon: np.ndarray,
                        n_samples: int = 2000) -> float:
    """
    Approximate Kendall's rank correlation between upper-triangular
    entries of D_clean and D_recon. Uses random subsampling for speed.
    """
    N = D_clean.shape[0]
    triu_idx = np.triu_indices(N, k=1)
    n_pairs = len(triu_idx[0])

    if n_pairs > n_samples:
        chosen = np.random.choice(n_pairs, size=n_samples, replace=False)
        i = triu_idx[0][chosen]
        j = triu_idx[1][chosen]
    else:
        i, j = triu_idx

    d1 = D_clean[i, j]
    d2 = D_recon[i, j]

    # Concordant/discordant
    n = len(d1)
    if n < 2:
        return 0.0

    # Compute all pairwise comparisons
    d1_diff = d1[:, np.newaxis] - d1[np.newaxis, :]  # (n, n)
    d2_diff = d2[:, np.newaxis] - d2[np.newaxis, :]

    # Signs
    s1 = np.sign(d1_diff)
    s2 = np.sign(d2_diff)

    # Concordant: same sign; discordant: opposite sign
    # Only consider upper triangle (pairs of pairs) to avoid double count
    triu2 = np.triu_indices(n, k=1)
    prod = s1[triu2] * s2[triu2]

    n_c = np.sum(prod > 0)
    n_d = np.sum(prod < 0)
    n_t = n_c + n_d

    if n_t == 0:
        return 0.0

    tau = (n_c - n_d) / n_t
    return float(tau)


def manifold_consistency(V_clean: np.ndarray, V_recon: np.ndarray,
                         n_neighbors: int = 10, metric: str = 'tau',
                         n_samples_tau: int = 2000) -> float:
    """
    Compute manifold consistency between clean and reconstructed data.

    Args:
        V_clean: (N, L) array of clean voltage curves.
        V_recon: (N, L) array of reconstructed voltage curves.
        n_neighbors: for neighbor purity.
        metric: 'tau' | 'neighbor_purity' | 'both'.
        n_samples_tau: max pairwise comparisons for Kendall's tau estimate.

    Returns:
        Scalar consistency metric.
    """
    V_clean = _to_2d(V_clean)
    V_recon = _to_2d(V_recon)
    N = V_clean.shape[0]

    if N < 2:
        return 0.0

    D_clean = _pairwise_distances(V_clean)
    D_recon = _pairwise_distances(V_recon)

    if metric == 'tau':
        return _kendall_tau_approx(D_clean, D_recon, n_samples=n_samples_tau)

    if metric == 'neighbor_purity':
        if N < n_neighbors + 1:
            return 0.0
        # k-NN in clean space
        np.fill_diagonal(D_clean, np.inf)
        clean_nn = np.argpartition(D_clean, n_neighbors, axis=1)[:, :n_neighbors]
        # k-NN in recon space
        np.fill_diagonal(D_recon, np.inf)
        recon_nn = np.argpartition(D_recon, n_neighbors, axis=1)[:, :n_neighbors]
        # Jaccard overlap per sample
        overlaps = np.zeros(N)
        for i in range(N):
            s_clean = set(clean_nn[i])
            s_recon = set(recon_nn[i])
            intersection = len(s_clean & s_recon)
            union = len(s_clean | s_recon)
            overlaps[i] = intersection / union if union > 0 else 0.0
        return float(overlaps.mean())

    if metric == 'both':
        tau = _kendall_tau_approx(D_clean, D_recon, n_samples=n_samples_tau)
        np.fill_diagonal(D_clean, np.inf)
        np.fill_diagonal(D_recon, np.inf)
        clean_nn = np.argpartition(D_clean, n_neighbors, axis=1)[:, :n_neighbors]
        recon_nn = np.argpartition(D_recon, n_neighbors, axis=1)[:, :n_neighbors]
        overlaps = np.zeros(N)
        for i in range(N):
            s_clean = set(clean_nn[i])
            s_recon = set(recon_nn[i])
            intersection = len(s_clean & s_recon)
            union = len(s_clean | s_recon)
            overlaps[i] = intersection / union if union > 0 else 0.0
        return tau, float(overlaps.mean())

    raise ValueError(f"Unknown metric: {metric}")


def _neighbor_rank_matrix(D: np.ndarray) -> np.ndarray:
    """
    Compute rank (1-indexed) matrix from distance matrix.
    ranks[i, j] = rank of point j in i's nearest-neighbor list (1 = closest, excluding self).
    Self gets rank 0.
    """
    np.fill_diagonal(D, np.inf)
    ranks = np.argsort(np.argsort(D, axis=1), axis=1) + 1  # 1-indexed
    np.fill_diagonal(ranks, 0)
    return ranks


def trustworthiness(V_ref: np.ndarray, V_eval: np.ndarray, k: int = 10) -> float:
    """
    Trustworthiness: measures how well the local neighborhood structure
    of the reference space is preserved in the evaluated space.

    T(k) = 1 - (2/(N*k*(2N-3k-1))) * sum_i sum_{j in U_i^k} (r(i,j) - k)

    where U_i^k = points in kNN of i in evaluated space but NOT in kNN of i in reference.
    r(i,j) = rank of j in reference space neighbor list of i.

    Range: (~0, 1], 1 = perfect preservation.

    Args:
        V_ref: (N, L) reference data (e.g., clean).
        V_eval: (N, L) evaluated data (e.g., reconstructed).
        k: number of neighbors.

    Returns:
        scalar trustworthiness in [0, 1].
    """
    V_ref = _to_2d(V_ref)
    V_eval = _to_2d(V_eval)
    N = V_ref.shape[0]

    if N < 2 or k >= N:
        return 0.0

    D_ref = _pairwise_distances(V_ref)
    D_eval = _pairwise_distances(V_eval)

    ranks_ref = _neighbor_rank_matrix(D_ref)  # (N, N), 1-indexed, 0 on diagonal

    # kNN sets
    np.fill_diagonal(D_eval, np.inf)
    nn_eval = np.argpartition(D_eval, k, axis=1)[:, :k]  # (N, k)

    np.fill_diagonal(D_ref, np.inf)
    nn_ref = np.argpartition(D_ref, k, axis=1)[:, :k]  # (N, k)

    # For each point i: find intrusions (in nn_eval but not in nn_ref)
    total_penalty = 0.0
    for i in range(N):
        ref_set = set(nn_ref[i])
        for j in nn_eval[i]:
            if j not in ref_set:
                # Intrusion: j is in eval's kNN but not in ref's kNN
                r_ij = ranks_ref[i, j]  # rank of j in ref space
                total_penalty += max(0, r_ij - k)

    max_penalty = N * k * (2 * N - 3 * k - 1) / 2.0
    if max_penalty <= 0:
        return 1.0

    t = 1.0 - (2.0 * total_penalty) / (N * k * (2 * N - 3 * k - 1))
    return float(np.clip(t, 0.0, 1.0))


def continuity(V_ref: np.ndarray, V_eval: np.ndarray, k: int = 10) -> float:
    """
    Continuity: measures whether points that are neighbors in the reference
    space remain neighbors in the evaluated space.

    C(k) = 1 - (2/(N*k*(2N-3k-1))) * sum_i sum_{j in V_i^k} (r̂(i,j) - k)

    where V_i^k = points in kNN of i in reference but NOT in kNN of i in evaluated.
    r̂(i,j) = rank of j in evaluated space neighbor list of i.

    Range: (~0, 1], 1 = perfect continuity.

    Args:
        V_ref: (N, L) reference data (e.g., clean).
        V_eval: (N, L) evaluated data (e.g., reconstructed).
        k: number of neighbors.

    Returns:
        scalar continuity in [0, 1].
    """
    V_ref = _to_2d(V_ref)
    V_eval = _to_2d(V_eval)
    N = V_ref.shape[0]

    if N < 2 or k >= N:
        return 0.0

    D_ref = _pairwise_distances(V_ref)
    D_eval = _pairwise_distances(V_eval)

    ranks_eval = _neighbor_rank_matrix(D_eval)  # (N, N), 1-indexed

    # kNN sets
    np.fill_diagonal(D_eval, np.inf)
    nn_eval = np.argpartition(D_eval, k, axis=1)[:, :k]

    np.fill_diagonal(D_ref, np.inf)
    nn_ref = np.argpartition(D_ref, k, axis=1)[:, :k]

    total_penalty = 0.0
    for i in range(N):
        eval_set = set(nn_eval[i])
        for j in nn_ref[i]:
            if j not in eval_set:
                # Missing: j is in ref's kNN but not in eval's kNN
                r̂_ij = ranks_eval[i, j]  # rank of j in eval space
                total_penalty += max(0, r̂_ij - k)

    max_penalty = N * k * (2 * N - 3 * k - 1) / 2.0
    if max_penalty <= 0:
        return 1.0

    c = 1.0 - (2.0 * total_penalty) / (N * k * (2 * N - 3 * k - 1))
    return float(np.clip(c, 0.0, 1.0))


def knn_degradation_consistency(V_clean: np.ndarray, V_corrupted: np.ndarray,
                                 V_recon: np.ndarray, k: int = 10) -> float:
    """
    kNN Degradation Consistency: measures how many of the kNN relationships
    degraded by corruption are restored by reconstruction.

    For each point i, let NN_clean_i be its k nearest neighbors in clean space.
    Let L_i = neighbors lost under corruption (in NN_clean_i but NOT in kNN of corrupted).
    Let R_i = neighbors restored by reconstruction (in L_i that ARE in kNN of recon).

    Consistency = mean(|R_i| / |L_i|) over all i, with |L_i| > 0.

    Range: [0, 1], 1 = all degraded neighbors restored.

    Args:
        V_clean: (N, L) clean data.
        V_corrupted: (N, L) corrupted data.
        V_recon: (N, L) reconstructed data.
        k: number of neighbors.

    Returns:
        scalar consistency in [0, 1].
    """
    V_clean = _to_2d(V_clean)
    V_corrupted = _to_2d(V_corrupted)
    V_recon = _to_2d(V_recon)
    N = V_clean.shape[0]

    if N < 2 or k >= N:
        return 0.0

    D_clean = _pairwise_distances(V_clean)
    D_corrupt = _pairwise_distances(V_corrupted)
    D_recon = _pairwise_distances(V_recon)

    # kNN sets
    np.fill_diagonal(D_clean, np.inf)
    nn_clean = np.argpartition(D_clean, k, axis=1)[:, :k]

    np.fill_diagonal(D_corrupt, np.inf)
    nn_corrupt = np.argpartition(D_corrupt, k, axis=1)[:, :k]

    np.fill_diagonal(D_recon, np.inf)
    nn_recon = np.argpartition(D_recon, k, axis=1)[:, :k]

    restore_rates = []
    for i in range(N):
        clean_set = set(nn_clean[i])
        corrupt_set = set(nn_corrupt[i])
        recon_set = set(nn_recon[i])

        # Lost under corruption
        lost = clean_set - corrupt_set
        if not lost:
            continue

        # Restored by reconstruction
        restored = lost & recon_set
        restore_rates.append(len(restored) / len(lost))

    return float(np.mean(restore_rates)) if restore_rates else 0.0
