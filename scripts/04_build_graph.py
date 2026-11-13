"""Step 4 - assemble the knowledge graph around the researcher embeddings.

  * SIMILAR_TO : each researcher's KNN_K nearest neighbours by cosine similarity
  * COAUTHOR   : joint works between two selected researchers (from step 2)
  * communities: k-means on the embeddings (k picked by cosine silhouette), labelled by
                 the paper topics most over-represented in each community; grouped into
                 <= 8 colour-coded areas
  * x, y       : 2-D UMAP projection of the 768-d embeddings (cosine)

Outputs (data/processed/):
  researchers_enriched.json  researchers + layout, community, neighbours, top papers
  communities.json           fine-grained communities (k-means)
  areas.json                 <= 8 colour-coded research areas, each a group of communities
  graph.json                 node-link JSON (researchers, institutions, topics, communities)
  graph.graphml              same graph for Gephi / Cytoscape / networkx
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import (AMBIGUOUS_ROBOTICS_TOPICS, COMMUNITY_K_RANGE, KNN_K,  # noqa: E402
                           PROC, ROBOTICS_TOPICS, SEED)

N_AREAS = 8        # colour-coded research areas (each groups several communities)
SPLIT_SILHOUETTE = 0.4  # split a community when its 2-way split is at least this well separated
SPLIT_MIN_SIZE = 12     # ... and both halves have at least this many researchers

# OpenAlex catch-all topics that say little about a group ("Topic Modeling" is where
# OpenAlex files most language-model papers, the Transformer paper included).
GENERIC_TOPICS = {"Advanced Neural Network Applications", "Neural Networks and Applications",
                  "Machine Learning and Data Classification", "Topic Modeling",
                  "Machine Learning and Algorithms", "Artificial Intelligence Applications"}
SUBFIELD_NAMES = {
    "Computer Vision and Pattern Recognition": "Computer Vision",
    "Computer Networks and Communications": "Networks & Communications",
    "Radiology, Nuclear Medicine and Imaging": "Medical Imaging",
    "Computer Graphics and Computer-Aided Design": "Computer Graphics",
    "Computational Theory and Mathematics": "Theory & Algorithms",
    "Hardware and Architecture": "Hardware & Architecture",
    "Control and Systems Engineering": "Control Systems",
}
HANDS_ON_ROBOTICS = set(ROBOTICS_TOPICS) - AMBIGUOUS_ROBOTICS_TOPICS
# Display names for OpenAlex topics whose names mislead: OpenAlex files all deep RL
# (Atari, Go, ...) under "Reinforcement Learning in Robotics".
TOPIC_DISPLAY = {"Reinforcement Learning in Robotics": "Reinforcement Learning"}
# Communities with at least this share of robotics-tagged members form their own area
# instead of being merged into whatever happens to be nearest.
ROBOTICS_COMMUNITY_SHARE = 0.4


def distinctive(group_topics: dict, k: int = 3, exclude=frozenset(), scores: bool = False) -> dict:
    """Topics most over-represented in each group: p(t|g) * log(p(t|g) / p(t))."""
    total = Counter()
    for c in group_topics.values():
        total.update(c)
    tot_sum = sum(total.values()) or 1
    out = {}
    for g, ct in group_topics.items():
        cs = sum(ct.values()) or 1
        score = {t: (w / cs) * math.log((w / cs) / (total[t] / tot_sum))
                 for t, w in ct.items() if w / cs > 0.02 and t not in exclude}
        best = sorted(score, key=score.get, reverse=True)[:k]
        out[g] = [(t, score[t], ct[t] / cs) for t in best] if scores else best
    return out


def short_topic(name: str) -> str:
    s = re.sub(r"^(Advanced|Various)\s+", "", TOPIC_DISPLAY.get(name, name))
    s = re.sub(r"(\s+(and\s+)?(Techniques|Applications|Methods|Algorithms|Research))+$", "", s)
    s = re.sub(r"\s+(and|in|for)$", "", s)
    return (s.strip(" ,") or name).replace(" and ", " & ")


def name_areas(area_sf: dict, area_topics: dict, sizes: dict, multi: dict) -> dict:
    """Name each area by its most distinctive subfield (robotics topics count as the
    pseudo-subfield 'Robotics'). Larger areas choose first; an area whose best subfield
    is taken, or covers < 15% of its papers, is named by its distinctive topics instead."""
    sf = distinctive(area_sf, k=4, scores=True)
    tp = distinctive(area_topics, k=2, exclude=GENERIC_TOPICS, scores=True)
    cands = sorted(((score * math.log1p(sizes[a]), a, name) for a, lst in sf.items()
                    for name, score, share in lst if share >= 0.15), reverse=True)
    names, taken = {}, set()
    for _, a, name in cands:
        if a not in names and name not in taken:
            names[a] = SUBFIELD_NAMES.get(name, name)
            taken.add(name)
    for a in area_sf:
        if a not in names:
            t = tp.get(a, [])
            if a in multi:  # spans several communities: name it after its two largest
                names[a] = multi[a]
                continue
            keep = [x for x in t if x[1] >= 0.5 * t[0][1]] if t else []
            names[a] = " · ".join(short_topic(x[0]) for x in keep) or f"Area {a + 1}"
    return names


def main():
    researchers = {r["id"]: r for r in json.loads((PROC / "researchers.json").read_text())}
    ids = pd.read_parquet(PROC / "researcher_ids.parquet").researcher_id.tolist()
    E = np.load(PROC / "researcher_embeddings.npy")
    papers = pd.read_parquet(PROC / "papers.parquet").set_index("paper_id")
    auth = pd.read_parquet(PROC / "authorships.parquet")
    co = pd.read_parquet(PROC / "coauthors.parquet")
    n = len(ids)
    idx = {r: i for i, r in enumerate(ids)}
    k = min(KNN_K, n - 1)

    # ---- similarity graph -------------------------------------------------------
    S = E @ E.T
    np.fill_diagonal(S, -1)
    nn = np.argsort(-S, axis=1)[:, :k]
    sim_edges = {}
    for i in range(n):
        for rank, j in enumerate(nn[i], 1):
            a, b = sorted((i, j))
            sim_edges[(a, b)] = max(sim_edges.get((a, b), 0), float(S[i, j]))
    # ---- communities: k-means on the unit vectors (~ cosine), best silhouette wins ----
    # (Louvain on the kNN graph was tried first; it lumped small, unrelated groups such
    #  as classical robotics and HPC into one community.)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    k_lo, k_hi = COMMUNITY_K_RANGE
    k_hi = max(2, min(k_hi, n // 10))
    best = None
    for kc in range(min(k_lo, k_hi), k_hi + 1):
        lab = KMeans(kc, n_init=10, random_state=SEED).fit_predict(E)
        sc = silhouette_score(E, lab, metric="cosine")
        if best is None or sc > best[0]:
            best = (sc, kc, lab)
    print(f"communities: k={best[1]} (cosine silhouette {best[0]:.3f})")
    # Refinement: k-means can lump two well-separated groups (e.g. robotics with HCI) when the
    # best k shifts. Split a community in two when its 2-way split is clearly separated.
    lab = best[2].copy()
    for c in range(best[1]):
        m = np.where(lab == c)[0]
        if len(m) < 2 * SPLIT_MIN_SIZE:
            continue
        sub = KMeans(2, n_init=10, random_state=SEED).fit_predict(E[m])
        if min(np.bincount(sub)) >= SPLIT_MIN_SIZE and silhouette_score(E[m], sub, metric="cosine") >= SPLIT_SILHOUETTE:
            lab[m[sub == 1]] = lab.max() + 1
    n_split = lab.max() + 1 - best[1]
    if n_split:
        print(f"refinement: split {n_split} communities that held two well-separated groups")
    order = np.argsort(-np.bincount(lab))                # community 0 = largest
    comm_of = np.argsort(order)[lab]
    big = [np.where(comm_of == ci)[0] for ci in range(lab.max() + 1)]

    # topic profile of each researcher from their own top papers (weighted by citations)
    top = auth[auth.in_top_list].join(papers[["topic", "topic_id", "subfield", "cited_by_count"]], on="paper_id")
    top_all = top[top.researcher_id.isin(idx)]      # all top papers (for each researcher's list)
    top = top[top.topic.notna() & top.researcher_id.isin(idx)]  # papers with a topic (for labels)
    top["area_sf"] = np.where(top.topic_id.isin(HANDS_ON_ROBOTICS), "Robotics", top.subfield)
    top["w"] = 1 + np.log1p(top.cited_by_count)
    r_topics = top.groupby(["researcher_id", "topic"]).w.sum()

    comm_topics = defaultdict(Counter)
    for (rid, t), w in r_topics.items():
        comm_topics[comm_of[idx[rid]]][t] += w
    comm_best = distinctive(comm_topics, exclude=GENERIC_TOPICS)

    # ---- research areas: robotics communities form one area; the rest are grouped by
    #      centroid similarity (cosine, average linkage) into the remaining colour slots ----
    from sklearn.cluster import AgglomerativeClustering
    n_comm = len(big)
    cent = np.stack([E[comm_of == ci].mean(0) for ci in range(n_comm)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    is_rob = np.array([researchers[ids[i]]["is_robotics"] for i in range(n)])
    rob_comms = [ci for ci in range(n_comm) if is_rob[comm_of == ci].mean() >= ROBOTICS_COMMUNITY_SHARE]
    rest = [ci for ci in range(n_comm) if ci not in rob_comms]
    n_rest = N_AREAS - (1 if rob_comms else 0)
    area_raw = np.zeros(n_comm, dtype=int)
    if len(rest) > n_rest:
        area_raw[rest] = AgglomerativeClustering(n_clusters=n_rest, metric="cosine",
                                                 linkage="average").fit_predict(cent[rest])
    else:
        area_raw[rest] = np.arange(len(rest))
    area_raw[rob_comms] = area_raw[rest].max() + 1 if rest else 0
    # number areas by size so colour slot 1 goes to the largest area
    sizes = Counter()
    for ci in range(n_comm):
        sizes[area_raw[ci]] += int((comm_of == ci).sum())
    remap = {a: i for i, (a, _) in enumerate(sizes.most_common())}
    area_of_comm = np.array([remap[a] for a in area_raw])
    area_of = area_of_comm[comm_of]
    area_topics = defaultdict(Counter)
    for ci, ct in comm_topics.items():
        area_topics[area_of_comm[ci]].update(ct)
    area_sf = defaultdict(Counter)
    for rid, s_, w in zip(top.researcher_id, top.area_sf, top.w):
        area_sf[area_of[idx[rid]]][s_] += w
    area_size = Counter(area_of.tolist())
    multi = {}
    for a in set(area_of_comm.tolist()):
        cs = sorted((ci for ci in range(n_comm) if area_of_comm[ci] == a), key=lambda ci: -(comm_of == ci).sum())
        if len(cs) > 1:
            multi[a] = " · ".join(short_topic(comm_best[ci][0]) for ci in cs[:2])
    area_names = name_areas(area_sf, area_topics, area_size, multi)
    area_best = distinctive(area_topics, exclude=GENERIC_TOPICS)

    communities = []
    for ci in range(n_comm):
        members = [ids[i] for i in range(n) if comm_of[i] == ci]
        communities.append({
            "id": ci, "size": len(members), "area": int(area_of_comm[ci]),
            "label": " · ".join(short_topic(t) for t in comm_best[ci][:2]) or f"Community {ci + 1}",
            "top_topics": [t for t, _ in comm_topics[ci].most_common(5)],
            "distinctive_topics": comm_best[ci],
            "robotics_members": sum(researchers[m]["is_robotics"] for m in members),
            "top_members": sorted(members, key=lambda m: -researchers[m]["cited_by_count"])[:5],
        })
    areas = []
    for ai in range(len(remap)):
        members = [ids[i] for i in range(n) if area_of[i] == ai]
        areas.append({
            "id": ai, "size": len(members),
            "label": area_names[ai],
            "distinctive_topics": area_best[ai],
            "communities": [c["id"] for c in communities if c["area"] == ai],
            "robotics_members": sum(researchers[m]["is_robotics"] for m in members),
            "top_members": sorted(members, key=lambda m: -researchers[m]["cited_by_count"])[:5],
        })

    # ---- 2-D layout ---------------------------------------------------------------
    if n >= 10:
        import umap
        xy = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.12, metric="cosine",
                       random_state=SEED).fit_transform(E)
    else:
        xy = np.random.default_rng(SEED).normal(size=(n, 2))
    xy = (xy - xy.mean(0)) / xy.std(0)

    # ---- enrich researchers ------------------------------------------------------------
    co_map = defaultdict(list)
    for a, b, w in co.itertuples(index=False):
        if a in idx and b in idx:
            co_map[a].append((b, int(w)))
            co_map[b].append((a, int(w)))
    enriched = []
    for i, rid in enumerate(ids):
        r = dict(researchers[rid])
        mine = top_all[top_all.researcher_id == rid].sort_values("cited_by_count", ascending=False)
        r.update({
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "community": int(comm_of[i]), "community_label": communities[comm_of[i]]["label"],
            "area": int(area_of[i]), "area_label": areas[area_of[i]]["label"],
            "similar": [{"id": ids[j], "score": round(float(S[i, j]), 4)} for j in nn[i]],
            "coauthors": [{"id": b, "joint_works": w} for b, w in sorted(co_map[rid], key=lambda x: -x[1])],
            "paper_topics": [t for t, _ in r_topics.get(rid, pd.Series(dtype=float))
                             .sort_values(ascending=False).head(6).items()] if rid in r_topics.index.get_level_values(0) else [],
            "top_papers": [
                {"id": pid, "title": papers.at[pid, "title"],
                 "year": int(papers.at[pid, "year"]) if pd.notna(papers.at[pid, "year"]) else None,
                 "cited_by_count": int(papers.at[pid, "cited_by_count"]),
                 "doi": papers.at[pid, "doi"] if isinstance(papers.at[pid, "doi"], str) else None}
                for pid in mine.paper_id.head(5)],
        })
        enriched.append(r)
    (PROC / "researchers_enriched.json").write_text(json.dumps(enriched, ensure_ascii=False, allow_nan=False))
    (PROC / "communities.json").write_text(json.dumps(communities, indent=1, ensure_ascii=False))
    (PROC / "areas.json").write_text(json.dumps(areas, indent=1, ensure_ascii=False))

    # ---- export knowledge graph (without the paper layer; Neo4j holds papers) ----------
    K = nx.MultiDiGraph()
    for r in enriched:
        K.add_node(r["id"], kind="Researcher", name=r["name"], cited_by_count=r["cited_by_count"],
                   h_index=r["h_index"] or 0, works_count=r["works_count"],
                   community=r["community"], community_label=r["community_label"],
                   area=r["area"], area_label=r["area_label"],
                   robotics_share=r["robotics_share"], is_robotics=r["is_robotics"],
                   x=r["x"], y=r["y"], orcid=r["orcid"] or "",
                   institution=(r["institution"] or {}).get("name") or "",
                   country=(r["institution"] or {}).get("country") or "")
        inst = r["institution"]
        if inst and inst["id"]:
            K.add_node(inst["id"], kind="Institution", name=inst["name"] or "", country=inst["country"] or "")
            K.add_edge(r["id"], inst["id"], key="AFFILIATED_WITH", type="AFFILIATED_WITH")
        for t in r["topics"]:
            K.add_node(t["id"], kind="Topic", name=t["name"], subfield=t["subfield"], field=t["field"])
            K.add_edge(r["id"], t["id"], key="WORKS_ON", type="WORKS_ON", weight=t["count"])
        cid = f"C{r['community']}"
        K.add_node(cid, kind="Community", name=r["community_label"])
        K.add_edge(r["id"], cid, key="IN_COMMUNITY", type="IN_COMMUNITY")
    for (a, b), w in sim_edges.items():
        K.add_edge(ids[a], ids[b], key="SIMILAR_TO", type="SIMILAR_TO", weight=round(w, 4))
    for a, b, w in co.itertuples(index=False):
        if a in idx and b in idx:
            K.add_edge(a, b, key="COAUTHOR", type="COAUTHOR", weight=int(w))
    nx.write_graphml(K, PROC / "graph.graphml")
    (PROC / "graph.json").write_text(json.dumps(nx.node_link_data(K, edges="links"), ensure_ascii=False))

    types = Counter(d["type"] for _, _, d in K.edges(data=True))
    kinds = Counter(d["kind"] for _, d in K.nodes(data=True))
    print(f"nodes: {dict(kinds)}")
    print(f"edges: {dict(types)}")
    print(f"areas ({len(areas)}) and their communities:")
    for a in areas:
        print(f"  AREA {a['id']} n={a['size']:<4} rob={a['robotics_members']:<3} {a['label']}")
        for ci in a["communities"]:
            c = communities[ci]
            print(f"      C{c['id']:<2} n={c['size']:<4} rob={c['robotics_members']:<3} {c['label']}")


if __name__ == "__main__":
    main()
