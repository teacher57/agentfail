"""Small, dependency-free stats for paired run comparison.

All tests are paired by task: each task contributes one value per run
(mean over its trials), and we test the per-task differences.
"""
from __future__ import annotations

import itertools
import random
from math import comb

EXACT_MAX = 16       # enumerate all 2^n sign flips up to this many nonzero diffs
N_PERM = 10_000
N_BOOT = 5_000


def pass_hat_k(successes_by_task: dict[str, list[bool]], k: int) -> float | None:
    """τ-bench pass^k: mean over tasks of C(c, k) / C(n, k); tasks with n < k are skipped."""
    vals = [comb(sum(t), k) / comb(len(t), k) for t in successes_by_task.values() if len(t) >= k]
    return sum(vals) / len(vals) if vals else None


def paired_permutation_p(diffs: list[float], seed: int = 0) -> float:
    """Two-sided sign-flip permutation test of mean(diffs) == 0."""
    d = [x for x in diffs if abs(x) > 1e-12]
    if not d:
        return 1.0
    obs = abs(sum(d)) - 1e-9
    if len(d) <= EXACT_MAX:
        hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs
                   for signs in itertools.product((1, -1), repeat=len(d)))
        return hits / 2 ** len(d)
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs for _ in range(N_PERM))
    return (hits + 1) / (N_PERM + 1)


def bootstrap_ci(diffs: list[float], alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(diffs), resampling tasks."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(N_BOOT))
    lo = means[int(alpha / 2 * N_BOOT)]
    hi = means[min(N_BOOT - 1, int((1 - alpha / 2) * N_BOOT))]
    return (lo, hi)
