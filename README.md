# backtest-validation

An anti-overfitting gate for strategy search. Search enough strategies and the luckiest one looks brilliant by chance; these
tools charge the best result for how many tries it took, so a "best of N" backtest cannot pass itself off as an edge.

## What's inside (`validation.py`, numpy + scipy)

| function | what it answers |
|---|---|
| `sharpe`, `probabilistic_sharpe_ratio` | Is this Sharpe ratio above a benchmark, given the sample length, skew and fat tails? |
| `deflated_sharpe_ratio`, `expected_max_sharpe` | Does the best strategy beat what the best of N *random* strategies would show? |
| `min_track_record_length` | How many observations before the Sharpe is believable? |
| `whites_reality_check`, `hansen_spa` | After searching L candidates, does any of them beat the benchmark? (stationary bootstrap) |
| `bonferroni`, `benjamini_hochberg` | Multiple-testing and false-discovery-rate control |
| `purged_walkforward`, `oos_gate` | Leakage-free walk-forward folds with a purge gap, and an out-of-sample gate |
| `evaluate_search` | All of the above on a (T x L) matrix of searched strategies' returns, with a one-line verdict |

```python
import numpy as np
from validation import evaluate_search

returns = np.random.default_rng(0).normal(0, 0.01, size=(1000, 200))   # 200 strategies of pure noise
print(evaluate_search(returns, n_boot=500)["verdict"])                  # NO EDGE (within data-snooping noise)
```

## The validator is validated

`python validation.py` runs a self-test that checks the gate the right way: by its false-positive rate over many independent
pure-noise searches, and by whether it still detects a real edge. On the current code:

```
(1) CALIBRATION -- 120 pure-noise searches, 80 strategies each:
    raw best Sharpe avg = 1.21 (looks like an edge every time)
    false-positive rate  DSR>0.95: 0%   RC_p<0.05: 8%   (target ~5%)
(2) POWER -- real edge (Sharpe~1.65) tested alone: DSR=1.000  RC_p=0.002  -> EDGE SURVIVES the search
(3) the same edge buried in 500 noise strategies: best Sharpe=1.65  DSR=0.982  RC_p=0.002  -> EDGE SURVIVES the search
```

Unit tests: `python -m unittest discover -s tests -v`

## References

- Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (2012) and "The Deflated Sharpe Ratio" (2014)
- White, "A Reality Check for Data Snooping", *Econometrica* (2000)
- Hansen, "A Test for Superior Predictive Ability", *Journal of Business & Economic Statistics* (2005)
- Politis & Romano, "The Stationary Bootstrap" (1994)
- Benjamini & Hochberg, "Controlling the False Discovery Rate" (1995)
- Lopez de Prado, *Advances in Financial Machine Learning* (2018), purged cross-validation

Built with AI coding tools under my direction; I designed the checks and verified the results. MIT license.
