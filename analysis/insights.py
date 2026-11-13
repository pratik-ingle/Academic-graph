"""Print a readable summary of the findings computed by analysis/export_findings.py.

All numbers are computed in one place (export_findings.py -> data/processed/findings.json);
this script only reports them, so the page and this summary cannot disagree.
Usage: .venv/bin/python analysis/insights.py   (run_all.sh saves it to analysis/insights_output.txt)
"""
import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
P = "data/processed/"
F = json.load(open(P + "findings.json"))
EN = json.load(open(P + "researchers_enriched.json"))
AL = {a["id"]: a["label"] for a in json.load(open(P + "areas.json"))}
CM = {c["id"]: c["label"] for c in json.load(open(P + "communities.json"))}
name = lambda i: EN[i]["name"]  # noqa: E731


def section(t):
    print(f"\n{'=' * 8} {t} {'=' * max(4, 70 - len(t))}")


print(f"Data retrieved {F.get('retrieved')}; graph fingerprint {F['fingerprint']}")

section("1. Are CS areas converging? (most-cited papers up to 2016 vs since 2022)")
c = F["convergence"]
print(f"{c['n']} researchers with >= {c['k']} papers in both periods and >= 5 in 2017-21.")
print("Change in area distinctness (same-area minus cross-area similarity):")
for d in c["distinct"]:
    extra = f"  (seeds {d['change_min']:+.0%}..{d['change_max']:+.0%})" if "change_min" in d else ""
    print(f"  {'fair  ' if d['fair'] else 'BIASED'} {d['method']:58s} {d['change']:+.0%}{extra}")
for key, role in (("toward_ai", "ai_area"), ("toward_rl", "rl_area")):
    if c.get(key):
        print(f"Shift toward the {AL[c[role]]} area relative to the average researcher (95% CI):")
        for r in sorted(c[key], key=lambda r: -r["value"]):
            print(f"  {AL[r['area']]:58s} n={r['n']:3d} {r['value']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")

section("2. LLM adoption")
L = F["llm"]
print(f"{L['researchers_all']:.0%} of {L['n_all']} researchers active since 2022 have >= 1 LLM paper "
      f"among their {L['per_researcher']} most-cited recent papers.")
for r in sorted(L["by_area"], key=lambda r: -r["researchers"]):
    print(f"  {AL[r['area']]:58s} {r['researchers']:5.0%} of {r['n']}")
if L["cs_population"]:
    print("LLM share of all computer-science papers:",
          ", ".join(f"{d['year']}{'*' if d['partial'] else ''} {d['share']:.2%}" for d in L["cs_population"]), "(* partial year)")

section("3. Robotics")
R = F["robotics"]
if R:
    print("Roboticists, shift toward RL relative to the average researcher; most toward:",
          ", ".join(f"{name(i)} ({d:+.3f})" for i, d in R["people"][:5]))
    print("... most away:", ", ".join(f"{name(i)} ({d:+.3f})" for i, d in R["people"][-5:]))

section("4. Collaboration vs similarity (co-authors' shared papers excluded)")
C = F["collab"]
h = C["homophily"]
print(f"Most vs least similar tenth of pairs: {h[9]['p'] / h[0]['p']:.0f}x as likely to have co-authored "
      f"({C['ratio_with_shared_papers']:.0f}x if shared papers are left in); {C['pairs_dropped']} pairs dropped.")
print("Observed / predicted co-author pairs per community (similarity only | + shared employer | strong ties):")
for r in sorted(C["communities"], key=lambda r: r["ratio"]):
    print(f"  {CM[r['community']][:56]:56s} {r['ratio']:5.2f} | {r['ratio_org']:5.2f} | {r['ratio_strong']:5.2f}  "
          f"same-employer pairs {r['same_org']:.1%}")
print(f"corr(same-employer share, ratio) = {C['corr_org']:.2f}")
print("Research twins (verified 0 joint works):", "; ".join(f"{name(i)} ~ {name(j)} {s:.2f}" for i, j, s in C["twins"]))

section("5. One-paper researchers")
O = F["onepaper"]
print(f">= 50% of citations from one paper: {O['count50']}; >= 70%: {O['count70']}")
for r in sorted(O["by_area"], key=lambda r: r["median"]):
    print(f"  {AL[r['area']]:58s} median {r['median']:4.0%}  one-paper {r['frac']:4.0%} of {r['n']}")
for s in O["sector"]:
    print(f"  {s['label']:58s} median {s['median']:4.0%}  one-paper {s['frac']:4.0%} of {s['n']}")
print("Largest cohorts:", "; ".join(f"{c['title'][:50]} ({len(c['members'])})" for c in O["cohorts"][:6]))
