"""Export the findings (see analysis/insights.py) as data/processed/findings.json for the
atlas page's Findings view. Researcher references are row indices into
researchers_enriched.json, the same order the page uses.

Usage: .venv/bin/python analysis/export_findings.py   (then scripts/06_build_viz.py)
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from agraph.openalex import OpenAlex  # noqa: E402
from analysis.common import area_roles, fingerprint, similarity_excluding_shared  # noqa: E402
from analysis.fetch_recent import PER as RECENT_PER  # noqa: E402

P = "data/processed/"
EN = json.load(open(P + "researchers_enriched.json"))
RAW = {r["id"]: r for r in json.load(open(P + "researchers.json"))}
CM = {c["id"]: c for c in json.load(open(P + "communities.json"))}
AL = {a["id"]: a["label"] for a in json.load(open(P + "areas.json"))}
ids = [r["id"] for r in EN]
idx = {r: i for i, r in enumerate(ids)}
n = len(ids)
area = np.array([r["area"] for r in EN])
comm = np.array([r["community"] for r in EN])
E = np.load(P + "researcher_embeddings.npy")
mu = np.load(P + "researcher_mean.npy")
S = E @ E.T
iu = np.triu_indices(n, 1)
rng = np.random.default_rng(0)
r3 = lambda v: round(float(v), 4)  # noqa: E731
out = {}

# ---- shared inputs -------------------------------------------------------------------
pe = np.load(P + "paper_embeddings.npy").astype(np.float32)
prow = {p: i for i, p in enumerate(pd.read_parquet(P + "paper_ids.parquet").paper_id)}
pp = pd.read_parquet(P + "papers.parquet").set_index("paper_id")
A = pd.read_parquet(P + "authorships.parquet")
A = A[A.in_top_list].join(pp[["year", "title", "cited_by_count"]], on="paper_id")
RP = pd.read_parquet(P + "recent_papers.parquet")
RE = np.load(P + "recent_embeddings.npy").astype(np.float32)
rrow = {p: i for i, p in enumerate(RP.paper_id)}
RA = pd.read_parquet(P + "recent_authorships.parquet")
now_rows = {r: [rrow[p] for p in g.paper_id] for r, g in RA.groupby("researcher_id")}
ROLES = area_roles(EN, json.load(open(P + "areas.json")), pp, pd.read_parquet(P + "authorships.parquet"))


def then_rows_for(cutoff):
    return {r: [prow[p] for p in g.paper_id] for r, g in A[A.year <= cutoff].groupby("researcher_id")}


def sample_centroids(rows, M, keep, K):
    V = np.stack([M[rng.choice(rows[r], K, replace=False)].mean(0) for r in keep])
    return V / np.linalg.norm(V, axis=1, keepdims=True)


# ---- 1 & 3. how research changed since 2016, measured without circularity -------------------
# Earlier versions compared the published areas (fitted mostly on pre-2016 papers) on raw
# vectors; that makes the past look more distinct by construction. Here:
#   * each period is summarised by equal random samples of papers (K per person, B repeats)
#   * each period is centred on its own mean, removing the shift every researcher shares
#   * areas used to test distinctness, and reference directions, come from 2017-21 papers,
#     which belong to neither period
#   * shifts are reported relative to the average researcher, with bootstrap intervals
K, B = 8, 10
norm = lambda V: V / np.linalg.norm(V, axis=1, keepdims=True)  # noqa: E731
centre = lambda V: norm(V - V.mean(0))  # noqa: E731
then_rows = then_rows_for(2016)
mid_rows = {r: [prow[p] for p in g.paper_id] for r, g in A[A.year.between(2017, 2021)].groupby("researcher_id")}
keep = [r for r in ids if len(then_rows.get(r, [])) >= K and len(now_rows.get(r, [])) >= K and len(mid_rows.get(r, [])) >= 5]
ka = area[[idx[r] for r in keep]]
T = norm(sum(sample_centroids(then_rows, pe, keep, K) for _ in range(B)))
N = norm(sum(sample_centroids(now_rows, RE, keep, K) for _ in range(B)))
M = norm(np.stack([pe[mid_rows[r]].mean(0) for r in keep]))
Tc, Nc, Mc = centre(T), centre(N), centre(M)
jj = np.triu_indices(len(keep), 1)


def gap(V, lab):
    sim = (V @ V.T)[jj]
    same = lab[jj[0]] == lab[jj[1]]
    return float(sim[same].mean() - sim[~same].mean())


n_areas = len(AL)
distinct = [
    {"method": "Published areas, raw vectors (earlier method)", "fair": False,
     "then": gap(T, ka), "now": gap(N, ka)},
    {"method": "Areas fitted on the up-to-2016 papers", "fair": False,
     **(lambda lab: {"then": gap(Tc, lab), "now": gap(Nc, lab)})(KMeans(n_areas, n_init=10, random_state=0).fit_predict(Tc))},
    {"method": "Areas fitted on the since-2022 papers", "fair": False,
     **(lambda lab: {"then": gap(Tc, lab), "now": gap(Nc, lab)})(KMeans(n_areas, n_init=10, random_state=0).fit_predict(Nc))},
]
for k_ in (n_areas - 2, n_areas, n_areas + 2):
    runs = []
    for seed in range(3):
        lab = KMeans(k_, n_init=10, random_state=seed).fit_predict(Mc)
        runs.append((gap(Tc, lab), gap(Nc, lab)))
    ch = [b / a - 1 for a, b in runs]
    distinct.append({"method": f"Neutral areas from 2017–21 papers, k = {k_} (3 seeds)", "fair": True,
                     "then": float(np.mean([a for a, _ in runs])), "now": float(np.mean([b for _, b in runs])),
                     "change_min": float(min(ch)), "change_max": float(max(ch))})
for d in distinct:
    d["change"] = d["now"] / d["then"] - 1
    d["then"], d["now"] = r3(d["then"]), r3(d["now"])


def relative_shift(role_area):
    """Per researcher: cos(now, ref) - cos(then, ref) minus the mean over everyone, where ref is
    the role area's 2017-21 centroid. 0 = moved like the average researcher."""
    if role_area is None:
        return None
    ref = Mc[ka == role_area].mean(0)
    ref /= np.linalg.norm(ref)
    d = Nc @ ref - Tc @ ref
    return d - d.mean()


def by_area(d, exclude):
    rows = []
    for a in AL:
        if a == exclude or not (ka == a).any():
            continue
        v = d[ka == a]
        bs = [rng.choice(v, len(v)).mean() for _ in range(4000)]
        rows.append({"area": int(a), "n": int(len(v)), "value": r3(v.mean()),
                     "ci": [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))]})
    return rows


shift_ai, shift_rl = relative_shift(ROLES["ai"]), relative_shift(ROLES["rl"])
si = [idx[r] for r in keep]
out["convergence"] = {
    "n": len(keep), "k": K, "distinct": distinct,
    "ai_area": ROLES["ai"], "rl_area": ROLES["rl"], "robotics_area": ROLES["robotics"],
    "toward_ai": by_area(shift_ai, ROLES["ai"]) if shift_ai is not None else None,
    "toward_rl": by_area(shift_rl, ROLES["rl"]) if shift_rl is not None else None,
    "shift_ai": {str(i): r3(v) for i, v in zip(si, shift_ai)} if shift_ai is not None else None,
    "shift_rl": {str(i): r3(v) for i, v in zip(si, shift_rl)} if shift_rl is not None else None,
}

# ---- 2. LLM adoption -----------------------------------------------------------------------
LLM = re.compile(r"\b(large language models?|LLMs?|GPT-?[34o]?\b|ChatGPT|foundation models?|in-context learning|"
                 r"instruction[- ]tun\w*|vision[- ]language[- ]action|chain[- ]of[- ]thought|multimodal large language)", re.I)
RP["llm"] = [bool(LLM.search(f"{t} {a}")) for t, a in zip(RP.title, RP.abstract.fillna(""))]
pp["llm"] = [bool(LLM.search(f"{t} {a}")) for t, a in zip(pp.title, pp.abstract.fillna(""))]
R2 = RA.join(RP.set_index("paper_id")[["llm", "year"]], on="paper_id")
R2 = R2[R2.researcher_id.isin(idx)]
R2["area"] = R2.researcher_id.map(lambda r: int(area[idx[r]]))
B2 = A.join(pp[["llm"]], on="paper_id")
B2 = B2[B2.researcher_id.isin(idx)]
B2["area"] = B2.researcher_id.map(lambda r: int(area[idx[r]]))
by_year = [{"year": int(y), "share": r3(g.llm.mean()), "n": int(len(g))} for y, g in R2.groupby("year") if y >= 2022]
llm_by_area = []
for a in AL:
    g, b = R2[R2.area == a], B2[(B2.area == a) & B2.year.between(2018, 2021)]
    per = g.groupby("researcher_id").llm.any()
    llm_by_area.append({"area": int(a), "base": r3(b.llm.mean()), "recent": r3(g.llm.mean()),
                    "researchers": r3(per.mean()), "n": int(len(per))})
per_r = R2.groupby("researcher_id").llm.agg(["mean", "size"])
share = {str(idx[r]): [r3(m), int(s)] for r, (m, s) in per_r.iterrows()}
examples = (RP[RP.llm].sort_values("cited_by_count", ascending=False).head(400)
            .merge(RA, on="paper_id").groupby("researcher_id").head(1))
ex = {str(idx[r]): t for r, t in zip(examples.researcher_id, examples.title) if r in idx}
# whole-field trend: LLM share of ALL computer-science papers per year (Beyond CS counts), which is
# not biased toward fast-cited topics the way a "most-cited recent papers" sample is
cs_pop = []
popf = Path("data/fields/population.json")
if popf.exists():
    popj = json.loads(popf.read_text())
    for y, c in sorted(popj["counts"].get("17", {}).items()):
        if "llm" in c and c.get("all"):
            cs_pop.append({"year": int(y), "share": r3(c["llm"] / c["all"]), "n": c["all"],
                           "partial": int(y) == popj.get("partial_year")})
out["llm"] = {"by_year": by_year, "by_area": llm_by_area, "share": share, "example": ex,
              "cs_population": cs_pop, "per_researcher": RECENT_PER,
              "base_all": r3(B2[B2.year.between(2018, 2021)].llm.mean()), "recent_all": r3(R2.llm.mean()),
              "researchers_all": r3(R2.groupby("researcher_id").llm.any().mean()),
              "n_all": int(R2.researcher_id.nunique())}

# ---- 3. robotics -------------------------------------------------------------------------------
rob, rl = ROLES["robotics"], ROLES["rl"]
if rob is not None and shift_rl is not None:
    people = sorted(((float(shift_rl[k]), si[k]) for k in range(len(keep)) if ka[k] == rob), reverse=True)
    out["robotics"] = {"area": int(rob), "rl_area": int(rl), "ai_area": ROLES["ai"],
                       "people": [[i, r3(d)] for d, i in people]}
else:
    out["robotics"] = None

# ---- 4. collaboration ----------------------------------------------------------------------
co = pd.read_parquet(P + "coauthors.parquet")
C = np.zeros((n, n))
for a, b, w in co.itertuples(index=False):
    if a in idx and b in idx:
        C[idx[a], idx[b]] = C[idx[b], idx[a]] = w
s_orig = S[iu]
S_adj = similarity_excluding_shared(EN, E, co)   # co-authors' shared papers left out
ok = ~np.isnan(S_adj[iu])
iu = (iu[0][ok], iu[1][ok])
s, y = S_adj[iu], (C[iu] > 0).astype(int)
y_orig = (C[np.triu_indices(n, 1)] > 0).astype(int)
qo = np.quantile(s_orig, [0.1, 0.9])
qs = np.quantile(s, np.linspace(0, 1, 11))
homophily = [{"decile": k + 1, "lo": r3(qs[k]), "hi": r3(qs[k + 1]),
              "p": float(y[(s >= qs[k]) & (s <= qs[k + 1])].mean())} for k in range(10)]


def org(name):
    if not name:
        return None
    low = name.lower()
    for k in ("google", "deepmind", "microsoft", "meta", "facebook", "openai", "nvidia", "ibm",
              "amazon", "tencent", "alibaba", "baidu", "huawei"):
        if k in low:
            return {"deepmind": "google", "facebook": "meta"}.get(k, k)
    return name


orgs = [org((r["institution"] or {}).get("name")) for r in EN]
industry = np.array([(r["institution"] or {}).get("type") == "company" for r in EN])
same_org = np.array([orgs[i] is not None and orgs[i] == orgs[j] for i, j in zip(*iu)], float)
X0 = np.c_[s, s ** 2, s ** 3]
X1 = np.c_[X0, same_org, same_org * s]
p0 = LogisticRegression(C=10, max_iter=3000).fit(X0, y).predict_proba(X0)[:, 1]
p1 = LogisticRegression(C=10, max_iter=3000).fit(X1, y).predict_proba(X1)[:, 1]
oa = OpenAlex()
cut = np.zeros(n)
for i, r in enumerate(EN):
    g = oa.get("/works", {"filter": f"author.id:{'|'.join(RAW[r['id']]['openalex_ids'])}",
                          "group_by": "authorships.author.id", "per-page": 200}).get("group_by", [])
    cut[i] = min(x["count"] for x in g) if len(g) >= 200 else 0
vis = np.minimum(cut[iu[0]], cut[iu[1]]) <= 2
ys = (C[iu] >= 3).astype(int)
ps = np.zeros(len(s))
ps[vis] = LogisticRegression(C=10, max_iter=3000).fit(X0[vis], ys[vis]).predict_proba(X0[vis])[:, 1]
same_c = comm[iu[0]] == comm[iu[1]]
communities = []
for c in CM:
    m = same_c & (comm[iu[0]] == c)
    mv = m & vis
    communities.append({"community": int(c), "n": int((comm == c).sum()),
                        "ratio": r3(y[m].sum() / p0[m].sum()), "ratio_org": r3(y[m].sum() / p1[m].sum()),
                        "ratio_strong": r3(ys[mv].sum() / ps[mv].sum()),
                        "same_org": r3(same_org[m].mean()), "industry": r3(industry[comm == c].mean())})
twins = []
for t in np.argsort(-s):
    i, j = iu[0][t], iu[1][t]
    if C[i, j] or orgs[i] == orgs[j]:
        continue
    f = (f"author.id:{'|'.join(RAW[ids[i]]['openalex_ids'])},"
         f"author.id:{'|'.join(RAW[ids[j]]['openalex_ids'])}")
    if oa.get("/works", {"filter": f, "per-page": 1, "select": "id"})["meta"]["count"] == 0:
        twins.append([int(i), int(j), r3(s[t])])
    if len(twins) == 6:
        break
out["collab"] = {"homophily": homophily, "communities": communities, "twins": twins,
                 "ratio_with_shared_papers": r3(y_orig[s_orig >= qo[1]].mean() / y_orig[s_orig <= qo[0]].mean()),
                 "pairs_dropped": int((~ok).sum()),
                 "corr_org": r3(np.corrcoef([c["same_org"] for c in communities],
                                            [c["ratio"] for c in communities])[0, 1]),
                 "same_org_overall": r3(same_org.mean())}

# ---- 5. one-paper researchers ------------------------------------------------------------
tot = {r["id"]: r["cited_by_count"] for r in EN}
top = A.sort_values("cited_by_count", ascending=False).groupby("researcher_id").head(1).copy()
top = top[top.researcher_id.isin(idx)]
top["share"] = top.cited_by_count / top.researcher_id.map(tot)
top["area"] = top.researcher_id.map(lambda r: int(area[idx[r]]))
top["industry"] = top.researcher_id.map(lambda r: bool(industry[idx[r]]))
onep = top[top.share >= 0.5]
cohorts = [{"title": t, "members": sorted(int(idx[r]) for r in g.researcher_id),
            "cites": int(g.cited_by_count.max())}
           for t, g in sorted(onep.groupby("title"), key=lambda kv: -len(kv[1]))[:8]]
out["onepaper"] = {
    "share": {str(idx[r]): [r3(sh), t] for r, sh, t in zip(top.researcher_id, top.share, top.title)},
    "by_area": [{"area": int(a), "median": r3(g.share.median()), "frac": r3((g.share >= .5).mean()), "n": int(len(g))}
                for a, g in top.groupby("area")],
    "sector": [{"label": lab, "median": r3(g.share.median()), "frac": r3((g.share >= .5).mean()), "n": int(len(g))}
               for lab, g in (("Industry", top[top.industry]), ("Academia & other", top[~top.industry]))],
    "count50": int((top.share >= .5).sum()), "count70": int((top.share >= .7).sum()), "cohorts": cohorts}

out["ids"] = ids
out["fingerprint"] = fingerprint()  # the page builder checks the findings match the current graph
meta = Path(P) / "meta.json"
out["retrieved"] = json.loads(meta.read_text())["retrieved"] if meta.exists() else None
with open(P + "findings.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, allow_nan=False)
print(f"findings.json written; {oa.stats()}")
print("twins:", [(EN[i]["name"], EN[j]["name"], v) for i, j, v in twins])
print("distinctness:", [(d["method"][:40], f"{d['change']:+.0%}") for d in out["convergence"]["distinct"]])
