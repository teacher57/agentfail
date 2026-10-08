"""Compare a candidate run against a baseline: pass rate, pass^k, per-category deltas, gate."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .classify import PRIORITY, SECONDARY, Result, classify
from .loader import Run
from .stats import bootstrap_ci, paired_permutation_p, pass_hat_k


@dataclass
class Delta:
    name: str
    base: float      # rate per sim (pass rate, or share of sims with this primary label)
    cand: float
    p: float         # paired permutation p-value over shared tasks
    ci: tuple[float, float] = (0.0, 0.0)

    @property
    def delta(self) -> float:
        return self.cand - self.base


@dataclass
class Comparison:
    base: Run
    cand: Run
    shared_tasks: list[str]
    only_base: list[str]
    only_cand: list[str]
    pass_rate: Delta
    pass_k: list[tuple[int, float | None, float | None]]
    categories: list[Delta]
    flips: dict[str, list[str]] = field(default_factory=dict)  # "fixed"/"broken" -> task ids


def _by_task(results: list[Result]) -> dict[str, list[Result]]:
    out: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        out[r.sim.task_id].append(r)
    return out


def _task_rate(rs: list[Result], pred) -> float:
    return sum(map(pred, rs)) / len(rs)


def _delta(name, base_t, cand_t, tasks, pred, with_ci=False) -> Delta:
    b = [_task_rate(base_t[t], pred) for t in tasks]
    c = [_task_rate(cand_t[t], pred) for t in tasks]
    diffs = [y - x for x, y in zip(b, c)]
    n = len(tasks) or 1
    return Delta(name, sum(b) / n, sum(c) / n, paired_permutation_p(diffs),
                 bootstrap_ci(diffs) if with_ci else (0.0, 0.0))


def compare(base: Run, cand: Run) -> Comparison:
    base_t = _by_task([classify(s) for s in base.sims])
    cand_t = _by_task([classify(s) for s in cand.sims])
    tasks = sorted(base_t.keys() & cand_t.keys(), key=lambda t: (len(t), t))

    passed = lambda r: r.sim.passed
    pass_rate = _delta("pass_rate", base_t, cand_t, tasks, passed, with_ci=True)

    succ_b = {t: [r.sim.passed for r in base_t[t]] for t in tasks}
    succ_c = {t: [r.sim.passed for r in cand_t[t]] for t in tasks}
    max_k = max(max(map(len, succ_b.values()), default=1), max(map(len, succ_c.values()), default=1))
    pass_k = [(k, pass_hat_k(succ_b, k), pass_hat_k(succ_c, k)) for k in range(1, max_k + 1)]

    cats = []
    for cat in PRIORITY:
        d = _delta(cat, base_t, cand_t, tasks, lambda r, c=cat: r.primary == c)
        if d.base or d.cand:
            cats.append(d)
    for cat in SECONDARY:
        d = _delta(cat, base_t, cand_t, tasks, lambda r, c=cat: c in r.labels)
        if d.base or d.cand:
            cats.append(d)

    flips = {"fixed": [], "broken": []}
    for t in tasks:
        b, c = _task_rate(base_t[t], passed), _task_rate(cand_t[t], passed)
        if b < 0.5 <= c:
            flips["fixed"].append(t)
        elif c < 0.5 <= b:
            flips["broken"].append(t)

    return Comparison(base, cand, tasks,
                      sorted(base_t.keys() - cand_t.keys()), sorted(cand_t.keys() - base_t.keys()),
                      pass_rate, pass_k, cats, flips)


def gate(cmp: Comparison, max_drop: float, alpha: float,
         category_limits: dict[str, float]) -> list[str]:
    """Return human-readable gate violations; empty list = pass.

    A violation needs both: the change exceeds its tolerance AND p < alpha
    (use alpha=1 to gate on the tolerance alone).
    """
    bad = []
    pr = cmp.pass_rate
    if -pr.delta > max_drop and (alpha >= 1 or pr.p < alpha):
        bad.append(f"pass_rate dropped {pr.delta:+.1%} (tolerance {max_drop:.1%}, p={pr.p:.3g})")
    by_name = {d.name: d for d in cmp.categories}
    for cat, limit in category_limits.items():
        d = by_name.get(cat)
        if d and d.delta > limit and (alpha >= 1 or d.p < alpha):
            bad.append(f"{cat} rose {d.delta:+.1%} of sims (tolerance {limit:.1%}, p={d.p:.3g})")
    return bad
