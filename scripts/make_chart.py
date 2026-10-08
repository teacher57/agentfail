"""Render the README chart: primary failure cause, baseline vs candidate, as a static SVG.

    python3 scripts/make_chart.py BASE.json CAND.json docs/failures.svg [--base-name base] [--cand-name aug]

Stdlib only. Bars are share of sims per primary category (paired by task);
significant deltas (p < 0.05) get a delta + p annotation. Light/dark via prefers-color-scheme.
"""
from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentfail.classify import PRIORITY  # noqa: E402
from agentfail.compare import compare  # noqa: E402
from agentfail.loader import load_tau2  # noqa: E402

W, LABEL_W, NOTE_W = 760, 150, 170
PLOT_W = W - LABEL_W - NOTE_W - 24
BAR_H, BAR_GAP, GROUP_GAP = 11, 2, 16
TOP = 92

STYLE = """
  .surface { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .t3 { fill: #75746f; }
  .grid { stroke: #e4e3df; } .axis { stroke: #a9a8a2; }
  .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; }
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; } .t1 { fill: #ffffff; } .t2 { fill: #c3c2b7; } .t3 { fill: #9a9990; }
    .grid { stroke: #2f2f2d; } .axis { stroke: #55554f; }
    .s1 { fill: #3987e5; } .s2 { fill: #d95926; }
  }
"""


def bar(x0, y, w, h, cls, r=4):
    """Bar anchored at x0 (square), rounded only at its data end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f'<path class="{cls}" d="M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} '
            f'v{h - 2 * r:.1f} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>')


def render(cmp, base_name, cand_name, alpha=0.05) -> str:
    cats = [d for d in cmp.categories if d.name in PRIORITY]
    vmax = max(max(d.base, d.cand) for d in cats)
    step = 0.05
    xmax = step * (int(vmax / step) + 1)
    sx = lambda v: LABEL_W + v / xmax * PLOT_W
    group_h = 2 * BAR_H + BAR_GAP
    H = TOP + len(cats) * (group_h + GROUP_GAP) + 56

    pr = cmp.pass_rate
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-labelledby="title desc">',
           f'<title id="title">Primary failure cause: {escape(base_name)} vs {escape(cand_name)}</title>',
           f'<desc id="desc">Share of simulations whose primary failure is each category. '
           f'Pass rate {pr.base:.1%} to {pr.cand:.1%}.</desc>',
           f"<style>{STYLE}</style>",
           f'<rect class="surface" width="{W}" height="{H}" rx="8"/>',
           f'<text class="t1" x="20" y="30" font-size="16" font-weight="600">'
           f'Why runs fail: {escape(base_name)} vs {escape(cand_name)}</text>',
           f'<text class="t2" x="20" y="50" font-size="12">Primary failure cause, share of sims '
           f'· {len(cmp.shared_tasks)} τ²-retail tasks · pass rate {pr.base:.1%} → {pr.cand:.1%} '
           f'(p={pr.p:.2f})</text>']

    # legend
    lx = 20
    for cls, name in (("s1", base_name), ("s2", cand_name)):
        out.append(f'<rect class="{cls}" x="{lx}" y="62" width="12" height="12" rx="3"/>')
        out.append(f'<text class="t2" x="{lx + 18}" y="72" font-size="12">{escape(name)}</text>')
        lx += 30 + 7 * len(name)

    # grid + x labels
    ybot = H - 50
    v = 0.0
    while v <= xmax + 1e-9:
        x = sx(v)
        out.append(f'<line class="{"axis" if v == 0 else "grid"}" x1="{x:.1f}" y1="{TOP - 6}" '
                   f'x2="{x:.1f}" y2="{ybot}" stroke-width="1"/>')
        out.append(f'<text class="t3" x="{x:.1f}" y="{ybot + 16}" font-size="11" '
                   f'text-anchor="middle">{v:.0%}</text>')
        v += step

    y = TOP
    for d in cats:
        sig = d.p < alpha
        out.append(f'<text class="t1" x="{LABEL_W - 10}" y="{y + group_h / 2 + 4:.1f}" font-size="12" '
                   f'text-anchor="end">{d.name}</text>')
        for i, (val, cls) in enumerate(((d.base, "s1"), (d.cand, "s2"))):
            by = y + i * (BAR_H + BAR_GAP)
            out.append(bar(sx(0), by, sx(val) - sx(0), BAR_H, cls))
            if sig:  # selective value labels: only where the change is significant
                out.append(f'<text class="t2" x="{sx(val) + 4:.1f}" y="{by + BAR_H - 2:.1f}" '
                           f'font-size="10">{val:.1%}</text>')
        if sig:
            p = "p<0.001" if d.p < 0.001 else f"p={d.p:.2f}"
            out.append(f'<text class="t1" x="{W - NOTE_W}" y="{y + group_h / 2 + 4:.1f}" font-size="12" '
                       f'font-weight="600">{f"{d.delta * 100:+.1f}".replace("-", "−")} pp <tspan class="t2" font-weight="400">'
                       f'· {escape(p)}</tspan></text>')
        y += group_h + GROUP_GAP

    out.append(f'<text class="t3" x="20" y="{H - 10}" font-size="10">Paired by task; p from a sign-flip '
               f'permutation test. Annotated rows: p &lt; {alpha}. Generated by agentfail.</text>')
    out.append("</svg>")
    return "\n".join(s for s in out if s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("out")
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--cand-name", default="candidate")
    a = ap.parse_args()
    cmp = compare(load_tau2(a.base), load_tau2(a.cand))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render(cmp, a.base_name, a.cand_name))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
