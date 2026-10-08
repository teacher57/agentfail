"""agentfail CLI.

    python -m agentfail classify RESULTS.json [--examples N] [--jsonl OUT]
    python -m agentfail compare BASE.json CAND.json [--max-drop X] [--alpha A]
                                [--max-increase CAT=X ...] [--json OUT]

`compare` exits 1 when the regression gate fails, so it can run in CI.
"""
from __future__ import annotations

import argparse
import json
import sys

from .classify import PRIORITY, SECONDARY, classify, summarize
from .compare import compare, gate
from .loader import load_tau2


def cmd_classify(a) -> int:
    run = load_tau2(a.path)
    results = [classify(s) for s in run.sims]
    s = summarize(results)
    n = s["n"]
    print(f"{run.path.name}  model={run.model}  sims={n}  pass={s['passed']} ({s['passed'] / n:.1%})")
    if s["fault"]:
        user = f" (simulated by {run.user_model})" if run.user_model else ""
        print(f"failures by fault: agent {s['fault']['agent']}, user {s['fault']['user']}{user}")
    print(f"\n{'category':<20}{'primary':>8}{'any(fail)':>11}{'any(pass)':>11}")
    for cat in PRIORITY + SECONDARY:
        p, f, ok = s["primary"][cat], s["any_failed"][cat], s["any_passed"][cat]
        if p or f or ok:
            print(f"{cat:<20}{p:>8}{f:>11}{ok:>11}")

    if a.examples:
        print()
        for cat in PRIORITY:
            ex = [r for r in results if r.primary == cat][: a.examples]
            for r in ex:
                d = next((f.detail for f in r.findings if f.label == cat), "")
                print(f"[{cat}] task {r.sim.task_id} t{r.sim.trial}: {d[:140]}")

    if a.jsonl:
        with open(a.jsonl, "w") as fh:
            for r in results:
                fh.write(json.dumps({
                    "task_id": r.sim.task_id, "trial": r.sim.trial, "reward": r.sim.reward,
                    "primary": r.primary,
                    "findings": [{"label": f.label, "detail": f.detail} for f in r.findings],
                }) + "\n")
        print(f"\nwrote {a.jsonl}")
    return 0


def _limit(s: str) -> tuple[str, float]:
    cat, _, val = s.partition("=")
    if cat not in PRIORITY + SECONDARY or not val:
        raise argparse.ArgumentTypeError(f"expected CATEGORY=RATE with a known category, got {s!r}")
    return cat, float(val)


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _star(p: float, alpha: float) -> str:
    return "*" if p < alpha else " "


def cmd_compare(a) -> int:
    base, cand = load_tau2(a.base), load_tau2(a.cand)
    cmp = compare(base, cand)
    pr = cmp.pass_rate
    print(f"base: {base.path.name} ({base.model}, {len(base.sims)} sims)")
    print(f"cand: {cand.path.name} ({cand.model}, {len(cand.sims)} sims)")
    print(f"paired on {len(cmp.shared_tasks)} shared tasks", end="")
    if cmp.only_base or cmp.only_cand:
        print(f"  (ignored: {len(cmp.only_base)} base-only, {len(cmp.only_cand)} cand-only)", end="")
    print("\n")

    print(f"pass rate  {pr.base:.1%} -> {pr.cand:.1%}  ({pr.delta:+.1%}, "
          f"95% CI [{pr.ci[0]:+.1%}, {pr.ci[1]:+.1%}], p={_p(pr.p)})")
    for k, b, c in cmp.pass_k:
        fmt = lambda v: f"{v:.1%}" if v is not None else "  n/a"
        print(f"pass^{k}     {fmt(b)} -> {fmt(c)}")
    print(f"tasks fixed {len(cmp.flips['fixed'])}, broken {len(cmp.flips['broken'])}")

    print(f"\n{'primary category':<20}{'base':>8}{'cand':>8}{'delta':>9}{'p':>8}")
    for d in cmp.categories:
        print(f"{d.name:<20}{d.base:>8.1%}{d.cand:>8.1%}{d.delta:>+9.1%}{_p(d.p):>7}{_star(d.p, a.alpha)}")
    print(f"(rates = share of sims; * = p < {a.alpha}; {SECONDARY[0]} counts any occurrence)")

    violations = gate(cmp, a.max_drop, a.alpha, dict(a.max_increase or []))
    print("\nGATE: " + ("FAIL\n  - " + "\n  - ".join(violations) if violations else "PASS"))

    if a.json:
        out = {
            "base": str(base.path), "cand": str(cand.path), "shared_tasks": len(cmp.shared_tasks),
            "pass_rate": {"base": pr.base, "cand": pr.cand, "delta": pr.delta, "p": pr.p, "ci95": pr.ci},
            "pass_k": [{"k": k, "base": b, "cand": c} for k, b, c in cmp.pass_k],
            "categories": [{"name": d.name, "base": d.base, "cand": d.cand, "delta": d.delta, "p": d.p}
                           for d in cmp.categories],
            "flips": cmp.flips,
            "gate": {"passed": not violations, "violations": violations},
        }
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
    return 1 if violations else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agentfail")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify", help="label failures in a τ² results JSON")
    c.add_argument("path")
    c.add_argument("--examples", type=int, default=0, help="print N examples per primary category")
    c.add_argument("--jsonl", help="write per-sim findings to this file")
    c.set_defaults(fn=cmd_classify)

    c = sub.add_parser("compare", help="compare a candidate run to a baseline; exit 1 on regression")
    c.add_argument("base")
    c.add_argument("cand")
    c.add_argument("--max-drop", type=float, default=0.0,
                   help="tolerated absolute pass-rate drop, e.g. 0.02 (default 0)")
    c.add_argument("--alpha", type=float, default=0.05,
                   help="significance level; a regression must also have p < alpha (1 = ignore p)")
    c.add_argument("--max-increase", type=_limit, action="append", metavar="CAT=RATE",
                   help="also fail if a category's share of sims rises by more than RATE (repeatable)")
    c.add_argument("--json", help="write the comparison as JSON to this file")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
