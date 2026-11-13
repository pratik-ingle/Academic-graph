"""Analyse data/fields/ (from fields_fetch.py): is each field moving toward AI?

Prints tables and writes data/fields/results.json.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

os.chdir(Path(__file__).resolve().parents[1])
D = Path("data/fields")
POP = json.loads((D / "population.json").read_text())
FIELDS = POP["fields"]
AI_RE = re.compile(r"\b(machine[- ]learning|deep[- ]learning|artificial intelligence|artificial neural networks?|"
                   r"convolutional neural|large language models?|random forests?|support vector machines?|"
                   r"graph neural networks?|chatgpt|generative ai)\b", re.I)
LLM_RE = re.compile(r"\b(large language models?|chatgpt|gpt-4|generative ai)\b", re.I)
rng = np.random.default_rng(0)


def tag(df):
    text = df.title.fillna("") + " " + df.abstract.fillna("")
    df["ai"] = text.str.contains(AI_RE.pattern, case=False, regex=True)
    df["llm"] = text.str.contains(LLM_RE.pattern, case=False, regex=True)
    return df


def boot_ci(x, n=2000):
    x = np.asarray(x)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


res = {"fields": FIELDS, "researchers": {}, "top": {}, "population": {}, "examples": {}}

# ---- 1. top researchers then vs now ------------------------------------------------------
RS = pd.read_parquet(D / "researchers.parquet")
RP = tag(pd.read_parquet(D / "researcher_papers.parquet"))
RP["cs"] = RP.field_id == "17"
g = RP.groupby(["researcher_id", "period"]).agg(n=("ai", "size"), ai=("ai", "mean"), llm=("llm", "any"),
                                                ai_any=("ai", "any"), cs=("cs", "mean")).unstack("period")
g.columns = [f"{a}_{b}" for a, b in g.columns]
g = g.join(RS.set_index("researcher_id")[["field_id", "name", "institution"]])
valid = g[(g.n_then >= 5) & (g.n_now >= 5)].copy()
valid["delta"] = valid.ai_now - valid.ai_then
print(f"{'field':44s} {'n':>4} {'AI share then':>13} {'now':>6} {'Δ (95% CI)':>22} {'p':>8} {'any AI now':>10} {'any LLM':>8} {'filed as CS then→now':>21}")
for fid, name in FIELDS.items():
    v = valid[valid.field_id == fid]
    if len(v) < 10:
        print(f"{name:44s} too few researchers ({len(v)})")
        continue
    lo, hi = boot_ci(v.delta)
    p = wilcoxon(v.ai_now, v.ai_then, zero_method="zsplit").pvalue if (v.delta != 0).any() else 1.0
    r = {"n": int(len(v)), "then": float(v.ai_then.mean()), "now": float(v.ai_now.mean()),
         "delta": float(v.delta.mean()), "ci": [lo, hi], "p": float(p),
         "any_then": float(v.ai_any_then.mean()), "any_now": float(v.ai_any_now.mean()),
         "llm_now": float(v.llm_now.mean()), "cs_then": float(v.cs_then.mean()), "cs_now": float(v.cs_now.mean()),
         "majority_ai_now": float((v.ai_now >= 0.5).mean()),
         "movers": [[n_, inst, float(d)] for n_, inst, d in
                    v.sort_values("delta", ascending=False)[["name", "institution", "delta"]].head(5).itertuples(index=False)]}
    res["researchers"][fid] = r
    print(f"{name:44s} {r['n']:4d} {r['then']:13.1%} {r['now']:6.1%} {r['delta']:+7.1%} [{lo:+.1%},{hi:+.1%}] {p:8.0e} "
          f"{r['any_now']:10.0%} {r['llm_now']:8.0%} {r['cs_then']:9.1%} → {r['cs_now']:.1%}")

# ---- 2. top-cited papers per year, 3. whole field --------------------------------------------
TP = tag(pd.read_parquet(D / "top_papers.parquet"))
print(f"\n{'AI share':44s} " + " ".join(f"{y:>6}" for y in range(2014, 2027)))
for fid, name in FIELDS.items():
    t = TP[TP.field == fid].groupby("pub_year").agg(ai=("ai", "mean"), llm=("llm", "mean"), n=("ai", "size"))
    cnt = POP["counts"][fid]
    pop_ai = {int(y): c["ai"] / c["all"] for y, c in cnt.items() if c.get("all")}
    pop_llm = {int(y): c["llm"] / c["all"] for y, c in cnt.items() if c.get("all") and "llm" in c}
    res["top"][fid] = {int(y): {"ai": float(r.ai), "llm": float(r.llm), "n": int(r.n)} for y, r in t.iterrows()}
    res["population"][fid] = {y: {"ai": pop_ai[y], "llm": pop_llm.get(y), "all": cnt[str(y)]["all"]} for y in pop_ai}
    print(f"{name[:30]:30s} top-100 " + " ".join(f"{t.ai.get(y, np.nan):6.0%}" for y in range(2014, 2027)))
    print(f"{'':30s} all     " + " ".join(f"{pop_ai.get(y, np.nan):6.1%}" for y in range(2014, 2027)))

print("\nAI 'citation premium' = AI share among the 100 most-cited papers / AI share among all papers")
print(f"{'field':44s} {'2015-17':>8} {'2022-24':>8}   {'all-paper AI share 2015→2024':>28}   {'LLM share of all 2024':>21}")
for fid, name in FIELDS.items():
    top, pop = res["top"][fid], res["population"][fid]
    prem = lambda ys: np.mean([top[y]["ai"] for y in ys]) / max(1e-9, np.mean([pop[y]["ai"] for y in ys]))  # noqa: E731
    res["population"][fid]["premium"] = {"2015-17": prem([2015, 2016, 2017]), "2022-24": prem([2022, 2023, 2024])}
    toa = POP.get("top_oa", {}).get(fid, {})
    if all(str(y) in toa for y in (2022, 2023, 2024)):
        # same detector (OpenAlex search) for the top-cited set and for all papers
        top_share = np.mean([toa[str(y)]["ai"] / toa[str(y)]["all"] for y in (2022, 2023, 2024)])
        res["population"][fid]["premium"]["2022-24 same detector"] = top_share / np.mean([pop[y]["ai"] for y in (2022, 2023, 2024)])
    print(f"{name:44s} {prem([2015, 2016, 2017]):7.1f}x {prem([2022, 2023, 2024]):7.1f}x   "
          f"{pop[2015]['ai']:12.2%} → {pop[2024]['ai']:.2%} ({pop[2024]['ai'] / max(pop[2015]['ai'], 1e-9):.0f}x)   {pop[2024]['llm'] or 0:12.2%}")

# examples: most-cited recent AI papers per field
for fid in FIELDS:
    ex = TP[(TP.field == fid) & TP.ai & (TP.pub_year >= 2022)].sort_values("cited_by_count", ascending=False).head(3)
    res["examples"][fid] = [[t, int(y), int(c)] for t, y, c in zip(ex.title, ex.pub_year, ex.cited_by_count)]
(D / "results.json").write_text(json.dumps(res, indent=1, default=float))
print("\nwrote data/fields/results.json")

# ---- robustness: strict rule (AI term in title, or >= 2 mentions in the abstract) ----------------
RP["strict"] = RP.title.fillna("").str.contains(AI_RE) | (RP.abstract.fillna("").str.count(AI_RE) >= 2)
g2 = RP.groupby(["researcher_id", "period"]).agg(n=("strict", "size"), s=("strict", "mean"), a=("strict", "any")).unstack("period")
g2.columns = [f"{a}_{b}" for a, b in g2.columns]
g2 = g2.join(RS.set_index("researcher_id").field_id)
v2 = g2[(g2.n_then >= 5) & (g2.n_now >= 5)]
print(f"\nStrict rule: {'field':36s} {'then':>6} {'now':>6} {'p':>8} {'any now':>8}")
for fid, name in FIELDS.items():
    x = v2[v2.field_id == fid]
    if fid not in res["researchers"] or len(x) < 10:
        continue
    p = wilcoxon(x.s_now, x.s_then, zero_method="zsplit").pvalue if (x.s_now != x.s_then).any() else 1.0
    res["researchers"][fid]["strict"] = {"then": float(x.s_then.mean()), "now": float(x.s_now.mean()), "p": float(p),
                                         "any_now": float(x.a_now.mean())}
    print(f"             {name[:36]:36s} {x.s_then.mean():6.1%} {x.s_now.mean():6.1%} {p:8.0e} {x.a_now.mean():8.0%}")

# ---- are top-cited AI papers about AI (term in title) or using it? --------------------------------
# AI as the subject: the AI terms above, plus the abbreviations that are unambiguous in a title
TITLE_RE = re.compile(r"\b(machine[- ]learning|deep[- ]learning|artificial intelligence|artificial neural networks?|"
                      r"convolutional neural|large language models?|random forests?|support vector machines?|"
                      r"graph neural networks?|chatgpt|generative ai|AI|LLMs?|GPT(-?\d\w*)?)\b", re.I)
recent = TP[TP.pub_year.between(2022, 2025) & TP.ai]
print(f"\nTop-cited AI papers 2022-25 with an AI term in the title: ")
for fid, name in FIELDS.items():
    t = recent[recent.field == fid]
    res["top"][fid]["about_ai"] = float(t.title.fillna("").str.contains(TITLE_RE).mean())
    print(f"  {name:44s} {res['top'][fid]['about_ai']:5.0%} of {len(t)}")
(D / "results.json").write_text(json.dumps(res, indent=1, default=float))
