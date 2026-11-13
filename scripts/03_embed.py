"""Step 3 - SPECTER2 embeddings for papers, then one vector per researcher.

Paper vector   = SPECTER2(title [SEP] abstract), proximity adapter, L2-normalised.
Researcher vec = citation-weighted mean of the paper vectors of their own top-cited
                 works, weight = 1 + ln(1 + citations), L2-normalised; then the mean
                 researcher vector is subtracted and the result re-normalised.
                 (log damping keeps one blockbuster paper from defining a whole career;
                 centring removes the direction every CS researcher shares)

Incremental: papers embedded in a previous run are reused.

Outputs (data/processed/):
  paper_embeddings.npy       float16 [n_papers, 768]  (row order = paper_ids.parquet)
  paper_ids.parquet
  researcher_embeddings.npy  float32 [n_researchers, 768] centred (row order = researcher_ids.parquet)
  researcher_embeddings_raw.npy, researcher_mean.npy  (before centring)
  researcher_ids.parquet
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import PROC  # noqa: E402
from agraph.embed import Specter2  # noqa: E402


def main():
    papers = pd.read_parquet(PROC / "papers.parquet")
    auth = pd.read_parquet(PROC / "authorships.parquet")
    researchers = json.loads((PROC / "researchers.json").read_text())
    fetched = set(auth.researcher_id)
    researchers = [r for r in researchers if r["id"] in fetched]

    # ---- paper embeddings (reuse previous run) -------------------------------
    # cache key = paper id + hash of the embedded text, so a changed abstract is re-embedded
    papers["text_hash"] = [hashlib.sha1(f"{t}\x1f{a or ''}".encode()).hexdigest()[:16]
                           for t, a in zip(papers.title, papers.abstract)]
    prev = {}
    pe, pi = PROC / "paper_embeddings.npy", PROC / "paper_ids.parquet"
    if pe.exists() and pi.exists():
        old = pd.read_parquet(pi)
        old_emb = np.load(pe)
        hashes = old.text_hash if "text_hash" in old else [None] * len(old)
        cur = dict(zip(papers.paper_id, papers.text_hash))
        prev = {pid: old_emb[i] for i, (pid, h) in enumerate(zip(old.paper_id, hashes))
                if h is None or cur.get(pid) == h}
    todo = papers[~papers.paper_id.isin(prev)]
    print(f"papers: {len(papers):,} total, {len(prev):,} cached, {len(todo):,} to embed")

    if len(todo):
        enc = Specter2("doc")
        texts = [enc.paper_text(t, a) for t, a in zip(todo.title, todo.abstract)]
        vecs = enc.encode(texts, progress=True)
        prev.update({pid: v for pid, v in zip(todo.paper_id, vecs)})

    ids = papers.paper_id.tolist()
    P = np.stack([prev[p] for p in ids]).astype(np.float32)
    np.save(pe, P.astype(np.float16))
    pd.DataFrame({"paper_id": ids, "text_hash": papers.text_hash.tolist()}).to_parquet(pi, index=False)

    # ---- researcher embeddings ------------------------------------------------
    row = {p: i for i, p in enumerate(ids)}
    cites = papers.set_index("paper_id").cited_by_count
    top = auth[auth.in_top_list]
    R = np.zeros((len(researchers), P.shape[1]), dtype=np.float32)
    n_papers = []
    for k, r in enumerate(researchers):
        pids = top.loc[top.researcher_id == r["id"], "paper_id"].tolist()
        w = np.array([1 + np.log1p(cites[p]) for p in pids], dtype=np.float32)
        v = (P[[row[p] for p in pids]] * w[:, None]).sum(0)
        R[k] = v / (np.linalg.norm(v) + 1e-12)
        n_papers.append(len(pids))
    # Averaged SPECTER2 vectors share a large common direction (mean pairwise cosine ~0.94).
    # Removing the mean researcher vector keeps what is distinctive about each person.
    mu = R.mean(0)
    Rc = R - mu
    Rc /= np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12
    np.save(PROC / "researcher_embeddings_raw.npy", R)
    np.save(PROC / "researcher_mean.npy", mu)
    np.save(PROC / "researcher_embeddings.npy", Rc)
    off = (R @ R.T)[~np.eye(len(R), dtype=bool)]
    offc = (Rc @ Rc.T)[~np.eye(len(R), dtype=bool)]
    print(f"pairwise cosine raw: mean {off.mean():.3f} [{off.min():.3f}, {off.max():.3f}]  "
          f"centred: mean {offc.mean():.3f} [{offc.min():.3f}, {offc.max():.3f}]")
    pd.DataFrame({"researcher_id": [r["id"] for r in researchers], "n_papers": n_papers}) \
        .to_parquet(PROC / "researcher_ids.parquet", index=False)
    print(f"researchers: {len(researchers)} vectors of dim {R.shape[1]} "
          f"(median {int(np.median(n_papers))} papers each)")


if __name__ == "__main__":
    main()
