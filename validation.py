#!/usr/bin/env python3
r"""validation.py -- an anti-overfitting gate for strategy search.

Raw "best of N backtests" is an overfitting factory: search enough strategies and the luckiest one looks
brilliant by pure chance. These tools penalize the best result for HOW MANY tries it took.

Implements (with references):
  - sharpe / probabilistic_sharpe_ratio (PSR) / deflated_sharpe_ratio (DSR) / min_track_record_length
      [Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" 2012; "Pseudo-Mathematics and Financial Charlatanism" 2014]
  - whites_reality_check + hansen_spa  -- stationary-bootstrap data-snooping tests
      [White, "A Reality Check for Data Snooping", Econometrica 2000; Hansen, "A Test for Superior Predictive Ability", 2005]
  - bonferroni / benjamini_hochberg (FDR)  -- multiple-testing corrections
  - purged_walkforward + oos_gate  -- leakage-free time-series cross-validation with a purge gap [Lopez de Prado, AFML]
  - evaluate_search()   -- the driver: a matrix of searched strategies' returns -> an honest verdict
  - __main__ self-test  -- shows pure-noise searches are rejected at about the nominal rate, and a real edge still passes

Convention: returns are per-period (e.g. daily); pass periods=252 to annualize Sharpe for display.
All significance math is done in per-period units internally.
"""
import sys
import numpy as np
from scipy import stats

__all__ = ["sharpe", "probabilistic_sharpe_ratio", "expected_max_sharpe", "deflated_sharpe_ratio", "min_track_record_length",
           "whites_reality_check", "hansen_spa", "bonferroni", "benjamini_hochberg", "purged_walkforward", "oos_gate", "evaluate_search"]

GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


def sharpe(returns, periods=252, annualized=True):
    r = np.asarray(returns, float); r = r[~np.isnan(r)]
    if r.size < 2 or r.std(ddof=1) == 0: return 0.0
    s = r.mean() / r.std(ddof=1)
    return float(s * np.sqrt(periods)) if annualized else float(s)


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, periods=252):
    """P(true Sharpe > sr_benchmark), accounting for sample length, skew and kurtosis.
    sr_benchmark is ANNUALIZED (converted internally). Bailey-Lopez de Prado."""
    r = np.asarray(returns, float); r = r[~np.isnan(r)]; n = r.size
    if n < 3 or r.std(ddof=1) == 0: return np.nan
    sr = r.mean() / r.std(ddof=1)                       # per-period
    sr_b = sr_benchmark / np.sqrt(periods)              # benchmark per-period
    g3 = stats.skew(r); g4 = stats.kurtosis(r, fisher=False)   # Pearson kurtosis (normal = 3)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    z = (sr - sr_b) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(trial_sr_var, n_trials):
    """E[max Sharpe] across n_trials INDEPENDENT null strategies, per-period units. Bailey-Lopez de Prado.
    trial_sr_var = variance of the per-period Sharpe ratios across the trials searched."""
    if n_trials < 2 or trial_sr_var <= 0: return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trial_sr_var) * ((1 - GAMMA) * z1 + GAMMA * z2))


def deflated_sharpe_ratio(returns, n_trials=None, trial_sr_var=None, all_trial_srs=None, periods=252):
    """DSR = PSR evaluated at the expected-max-Sharpe-under-the-null benchmark.
    DSR > 0.95 means the strategy beats what the best of n_trials random tries would produce.
    Supply all_trial_srs (annualized Sharpes of every searched strategy) OR (n_trials and trial_sr_var)."""
    if all_trial_srs is not None:
        srs_pp = np.asarray(all_trial_srs, float) / np.sqrt(periods)
        n_trials = srs_pp.size
        trial_sr_var = float(np.var(srs_pp, ddof=1)) if n_trials > 1 else 0.0   # one trial = no deflation
    sr0_pp = expected_max_sharpe(trial_sr_var, n_trials)           # per-period
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0_pp * np.sqrt(periods), periods=periods)


def min_track_record_length(returns, sr_benchmark=0.0, prob=0.95, periods=252):
    """Minimum number of observations for the Sharpe to exceed sr_benchmark with confidence `prob`."""
    r = np.asarray(returns, float); r = r[~np.isnan(r)]; n = r.size
    if n < 3 or r.std(ddof=1) == 0: return np.nan
    sr = r.mean() / r.std(ddof=1); sr_b = sr_benchmark / np.sqrt(periods)
    if sr <= sr_b: return np.inf
    g3 = stats.skew(r); g4 = stats.kurtosis(r, fisher=False)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (stats.norm.ppf(prob) / (sr - sr_b)) ** 2)


def _stationary_bootstrap_idx(n, mean_block, rng):
    """Politis-Romano stationary bootstrap indices (vectorized): geometric blocks with mean length mean_block."""
    p = 1.0 / max(1.0, mean_block)
    starts = rng.random(n) < p; starts[0] = True
    block_id = np.cumsum(starts) - 1
    pos = np.arange(n); start_pos = pos[starts]
    base = rng.integers(0, n, size=start_pos.size)        # a random origin per block
    return (base[block_id] + (pos - start_pos[block_id])) % n


def whites_reality_check(excess, mean_block=10, n_boot=2000, seed=0):
    """White's Reality Check. excess: (T, L), each strategy's per-period return MINUS the benchmark.
    H0: the best strategy does NOT beat the benchmark. A small p means some strategy beats it AFTER
    accounting for the full search of L candidates."""
    X = np.asarray(excess, float); T, L = X.shape
    fbar = X.mean(0); Vmax = (np.sqrt(T) * fbar).max()
    rng = np.random.default_rng(seed); count = 0
    for _ in range(n_boot):
        Xb = X[_stationary_bootstrap_idx(T, mean_block, rng)]
        if (np.sqrt(T) * (Xb.mean(0) - fbar)).max() >= Vmax: count += 1   # recentred under H0
    return (count + 1) / (n_boot + 1)


def hansen_spa(excess, mean_block=10, n_boot=2000, seed=0):
    """Hansen's SPA (consistent p-value): studentized, and drops hopeless strategies from the null,
    so a swarm of bad candidates cannot mask a good one (White's Reality Check is conservative there)."""
    X = np.asarray(excess, float); T, L = X.shape
    fbar = X.mean(0); omega = X.std(0, ddof=1); omega[omega == 0] = np.inf
    Tmax = max((np.sqrt(T) * fbar / omega).max(), 0.0)
    thresh = -np.sqrt(2 * np.log(max(np.log(T), 1.0001)))            # Hansen's A_n rate
    g = np.where(np.sqrt(T) * fbar / omega >= thresh, fbar, 0.0)     # consistent recentring
    rng = np.random.default_rng(seed); count = 0
    for _ in range(n_boot):
        Xb = X[_stationary_bootstrap_idx(T, mean_block, rng)]
        if max((np.sqrt(T) * (Xb.mean(0) - g) / omega).max(), 0.0) >= Tmax: count += 1
    return (count + 1) / (n_boot + 1)


def bonferroni(pvals, alpha=0.05):
    p = np.asarray(pvals, float); return p <= alpha / p.size


def benjamini_hochberg(pvals, alpha=0.05):
    """False-discovery-rate control. Returns a boolean mask of which hypotheses pass."""
    p = np.asarray(pvals, float); m = p.size; order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, m + 1) / m
    out = np.zeros(m, bool)
    if passed.any(): out[order[:np.where(passed)[0].max() + 1]] = True
    return out


def purged_walkforward(n, n_splits=5, purge=0):
    """Expanding-window walk-forward with a PURGE gap. Training data always ends strictly before the test fold,
    separated by `purge` observations (set it to at least the strategy's holding horizon), so no embargo is needed:
    post-test data can never enter a training window in an expanding design. Yields (train_idx, test_idx) and skips
    degenerate folds."""
    fold = n // (n_splits + 1)
    if fold < 1:
        return
    for i in range(1, n_splits + 1):
        ts = fold * i; te = fold * (i + 1) if i < n_splits else n
        train_end = max(0, ts - purge)
        if train_end <= 0 or te <= ts:               # skip empty folds (an empty fold's Sharpe of 0 would pollute the average)
            continue
        yield np.arange(0, train_end), np.arange(ts, te)


def oos_gate(returns, n_splits=5, purge=0, mode="median", periods=252):
    """Out-of-sample robustness gate on ONE strategy's per-period returns, via purged walk-forward.
    mode: 'all'    = every test fold's Sharpe > 0 (strict, and low power: it kills regime-conditional edges),
          'median' = the median test fold > 0,
          float    = that FRACTION of folds must be > 0 (e.g. 0.8 = 4 of 5).
    Returns (passed, test_fold_sharpes). The 'median' and fraction modes have a high false-positive rate on their own:
    use them together with deflated_sharpe_ratio > 0.95, never alone."""
    srs = [sharpe(returns[te], periods) for _, te in purged_walkforward(len(returns), n_splits, purge)]
    if not srs:
        return False, srs
    if mode == "all":
        passed = all(s > 0 for s in srs)
    elif mode == "median":
        passed = float(np.median(srs)) > 0
    else:
        passed = (sum(s > 0 for s in srs) / len(srs)) >= float(mode)
    return bool(passed), srs


def evaluate_search(returns_matrix, periods=252, benchmark=None, mean_block=10, n_boot=2000,
                    fdr_alpha=0.05, seed=0):
    """THE DRIVER. returns_matrix: (T, L) per-period returns of L searched strategies.
    Returns an honest verdict: the best raw Sharpe, its Deflated Sharpe, White's Reality Check and Hansen's SPA
    p-values, and how many strategies survive FDR control -- so 'best of N' cannot fool you."""
    R = np.asarray(returns_matrix, float); T, L = R.shape
    srs = np.array([sharpe(R[:, l], periods) for l in range(L)])
    best = int(np.nanargmax(srs))
    pvals = np.empty(L)                              # a NaN PSR (degenerate strategy) -> p = 1 (no evidence)
    for l in range(L):
        psr = probabilistic_sharpe_ratio(R[:, l], 0, periods)
        pvals[l] = (1.0 - psr) if np.isfinite(psr) else 1.0
    dsr = deflated_sharpe_ratio(R[:, best], all_trial_srs=srs, periods=periods)
    bench = np.zeros(T) if benchmark is None else np.asarray(benchmark, float)
    excess = R - bench[:, None]
    rc = whites_reality_check(excess, mean_block, n_boot, seed)
    spa = hansen_spa(excess, mean_block, n_boot, seed)
    fdr = benjamini_hochberg(pvals, fdr_alpha)
    survives = (dsr is not None and dsr > 0.95) and (rc < 0.05)
    return dict(n_strategies=L, n_obs=T, best_idx=best, best_sharpe=round(float(srs[best]), 3),
                best_DSR=round(float(dsr), 3), reality_check_p=round(float(rc), 4),
                spa_p=round(float(spa), 4), n_survive_fdr=int(fdr.sum()),
                verdict=("EDGE SURVIVES the search" if survives else "NO EDGE (within data-snooping noise)"))


def _selftest():
    """Validate the validator: a test is correct if its FALSE-POSITIVE RATE over many independent null draws is near
    the nominal level (~5%) -- not whether one arbitrary seed passes. Then show it still has the POWER to detect a real edge."""
    print("=== validation.py self-test ===")

    # (1) CALIBRATION -- over many pure-noise searches the false-positive rate must stay near 5%,
    #     even though the raw 'best of L' looks brilliant every single time.
    M, T, L = 120, 1000, 80
    base = np.random.default_rng(0); best_raw = []; fp_dsr = fp_rc = 0
    for m in range(M):
        R = base.normal(0, 0.01, size=(T, L))
        srs = np.array([sharpe(R[:, l]) for l in range(L)])
        b = int(np.nanargmax(srs)); best_raw.append(srs[b])
        if deflated_sharpe_ratio(R[:, b], all_trial_srs=srs) > 0.95: fp_dsr += 1
        if whites_reality_check(R, mean_block=5, n_boot=200, seed=m) < 0.05: fp_rc += 1
    print(f"(1) CALIBRATION -- {M} pure-noise searches, {L} strategies each:")
    print(f"    raw best Sharpe avg = {np.mean(best_raw):.2f} (looks like an edge every time)")
    print(f"    false-positive rate  DSR>0.95: {100*fp_dsr/M:.0f}%   RC_p<0.05: {100*fp_rc/M:.0f}%   (target ~5%)")

    # (2) POWER -- a genuine edge (annual Sharpe ~1.2), tested ALONE, must pass.
    rng = np.random.default_rng(3); T2 = 2500
    edge = rng.normal(1.2 / np.sqrt(252) * 0.01, 0.01, size=(T2, 1))
    r = evaluate_search(edge, n_boot=500, seed=1)
    print(f"(2) POWER -- real edge (Sharpe~{sharpe(edge[:,0]):.2f}) tested alone: "
          f"DSR={r['best_DSR']:.3f}  RC_p={r['reality_check_p']:.3f}  -> {r['verdict']}")

    # (3) the SAME edge buried in a 500-strategy search -- the search tax can swamp it.
    mixed = np.hstack([rng.normal(0, 0.01, size=(T2, 500)), edge])
    r2 = evaluate_search(mixed, n_boot=500, seed=1)
    print(f"(3) the same edge buried in 500 noise strategies: best Sharpe={r2['best_sharpe']:.2f}  "
          f"DSR={r2['best_DSR']:.3f}  RC_p={r2['reality_check_p']:.3f}  -> {r2['verdict']}")
    print("    Lesson: the more you search, the stronger the edge must be. Pre-register, then confirm out of sample.")

    ok = (fp_dsr / M <= 0.10) and (fp_rc / M <= 0.10) and r["verdict"].startswith("EDGE")
    print(f"\n{'ALL PASS: the gate is calibrated (~5% false positives) and has power' if ok else '*** CALIBRATION / POWER FAILURE ***'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
