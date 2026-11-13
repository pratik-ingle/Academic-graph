"""Helpers shared by analysis/insights.py and analysis/export_findings.py."""
import numpy as np
import pandas as pd

P = "data/processed/"
RL_TOPIC = "T10462"  # OpenAlex "Reinforcement Learning in Robotics" (in practice: all deep RL)


def area_roles(EN, areas_json, papers, authorships):
    """Identify the Robotics, RL and AI areas from the data rather than from label text,
    so a rerun that renames clusters cannot silently break the findings.
      robotics: highest share of robotics-tagged researchers
      rl:       highest share of top papers on the deep-RL topic (excluding robotics)
      ai:       highest share of top papers in the 'Artificial Intelligence' subfield
                (excluding robotics and rl)"""
    area = {r["id"]: r["area"] for r in EN}
    ids = [a["id"] for a in areas_json]
    rob_share = {a: np.mean([r["is_robotics"] for r in EN if r["area"] == a]) for a in ids}
    robotics = max(ids, key=rob_share.get)
    top = authorships[authorships.in_top_list].join(papers[["topic_id", "subfield"]], on="paper_id")
    top = top[top.researcher_id.isin(area)]
    top["area"] = top.researcher_id.map(area)
    by = top.groupby("area")
    rl_share = by.topic_id.apply(lambda s: (s == RL_TOPIC).mean())
    ai_share = by.subfield.apply(lambda s: (s == "Artificial Intelligence").mean())
    rl = max((a for a in ids if a != robotics), key=lambda a: rl_share.get(a, 0))
    ai = max((a for a in ids if a not in (robotics, rl)), key=lambda a: ai_share.get(a, 0))
    # a role only counts when the data clearly show it (otherwise None: the finding is skipped)
    return {"robotics": int(robotics) if rob_share[robotics] >= 0.4 else None,
            "rl": int(rl) if rl_share.get(rl, 0) >= 0.15 else None,
            "ai": int(ai) if ai_share.get(ai, 0) >= 0.5 else None}


def similarity_excluding_shared(EN, E, coauthors, min_papers=5):
    """Researcher similarity where co-authors' shared papers are left out of both profiles.

    Co-authors share papers, which makes their profiles similar by construction and would
    inflate any 'similar people collaborate' result. For every co-author pair both profiles
    are rebuilt from their own top papers minus the shared ones (same weighting and centring
    as scripts/03_embed.py). Pairs left with < min_papers papers get NaN."""
    ids = [r["id"] for r in EN]
    idx = {r: i for i, r in enumerate(ids)}
    mu = np.load(P + "researcher_mean.npy")
    pe = np.load(P + "paper_embeddings.npy").astype(np.float32)
    prow = {p: i for i, p in enumerate(pd.read_parquet(P + "paper_ids.parquet").paper_id)}
    cites = pd.read_parquet(P + "papers.parquet").set_index("paper_id").cited_by_count
    A = pd.read_parquet(P + "authorships.parquet")
    top = {r: list(g.paper_id) for r, g in A[A.in_top_list].groupby("researcher_id")}
    allp = {r: set(g.paper_id) for r, g in A.groupby("researcher_id")}

    def vec(papers):
        if len(papers) < min_papers:
            return None
        w = np.array([1 + np.log1p(cites[p]) for p in papers], np.float32)
        v = (pe[[prow[p] for p in papers]] * w[:, None]).sum(0)
        v = v / np.linalg.norm(v) - mu
        return v / np.linalg.norm(v)

    S = E @ E.T
    for a, b, _ in coauthors.itertuples(index=False):
        if a not in idx or b not in idx:
            continue
        shared = allp.get(a, set()) & allp.get(b, set())
        if not shared:
            continue
        va = vec([p for p in top.get(a, []) if p not in shared])
        vb = vec([p for p in top.get(b, []) if p not in shared])
        i, j = idx[a], idx[b]
        S[i, j] = S[j, i] = np.nan if va is None or vb is None else float(va @ vb)
    return S


def fingerprint(proc=P):
    """Identifies the graph the findings were computed on (researchers, areas, communities and
    profile vectors). scripts/06_build_viz.py refuses findings whose fingerprint differs."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha1()
    for f in ("researcher_ids.parquet", "areas.json", "communities.json", "researcher_embeddings.npy"):
        h.update((Path(proc) / f).read_bytes())
    return h.hexdigest()[:16]
