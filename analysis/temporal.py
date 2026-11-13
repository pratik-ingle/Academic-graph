"""Then (most-cited papers <= 2016) vs now (most-cited papers >= 2022) per researcher."""
import json, re
import numpy as np, pandas as pd
P = "data/processed/"
EN = json.load(open(P + "researchers_enriched.json")); AR = json.load(open(P + "areas.json"))
ids = [r["id"] for r in EN]; idx = {r: i for i, r in enumerate(ids)}
area = np.array([r["area"] for r in EN]); alab = {a["id"]: a["label"] for a in AR}
# classic papers
pe = np.load(P + "paper_embeddings.npy").astype(np.float32); pid = pd.read_parquet(P + "paper_ids.parquet").paper_id.tolist()
pp = pd.read_parquet(P + "papers.parquet").set_index("paper_id"); prow = {p: i for i, p in enumerate(pid)}
A = pd.read_parquet(P + "authorships.parquet"); A = A[A.in_top_list].join(pp[["year"]], on="paper_id")
then_rows = {r: [prow[p] for p in g.paper_id] for r, g in A[A.year <= 2016].groupby("researcher_id")}
# recent papers
RP = pd.read_parquet(P + "recent_papers.parquet"); RE = np.load(P + "recent_embeddings.npy").astype(np.float32)
rrow = {p: i for i, p in enumerate(RP.paper_id)}
RA = pd.read_parquet(P + "recent_authorships.parquet")
now_rows = {r: [rrow[p] for p in g.paper_id] for r, g in RA.groupby("researcher_id")}
K, B = 8, 20
keep = [r for r in ids if len(then_rows.get(r, [])) >= K and len(now_rows.get(r, [])) >= K]
ki = np.array([idx[r] for r in keep])
rng = np.random.default_rng(0)
def centroids(rows, M):
    out = np.zeros((len(keep), M.shape[1]), np.float32)
    for k, r in enumerate(keep):
        v = M[rng.choice(rows[r], K, replace=False)].mean(0); out[k] = v / np.linalg.norm(v)
    return out
