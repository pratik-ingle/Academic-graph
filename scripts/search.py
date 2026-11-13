"""Semantic search over the researcher embeddings.

  python scripts/search.py "legged locomotion with reinforcement learning"
  python scripts/search.py "graph neural networks" --neo4j        # use Neo4j vector indexes
  python scripts/search.py --like "Sergey Levine"                  # researchers most similar to someone

Text queries are embedded with the SPECTER2 ad-hoc query adapter, which shares a
vector space with the paper/researcher (proximity) embeddings.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, PROC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--like", help="researcher name: list the most similar researchers")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--neo4j", action="store_true")
    a = ap.parse_args()
    if not (a.query or a.like):
        ap.error("give a text query or --like NAME")

    R = {r["id"]: r for r in json.loads((PROC / "researchers_enriched.json").read_text())}
    r_ids = pd.read_parquet(PROC / "researcher_ids.parquet").researcher_id.tolist()
    E = np.load(PROC / "researcher_embeddings.npy")

    if a.like:
        hits = [r for r in R.values() if a.like.lower() in r["name"].lower()]
        if not hits:
            sys.exit(f"no researcher matching {a.like!r}")
        me = hits[0]
        q = E[r_ids.index(me["id"])]
        print(f"Most similar to {me['name']} ({(me['institution'] or {}).get('name')}):")
    else:
        from agraph.embed import Specter2
        q_raw = Specter2("query").encode([a.query])[0]
        # researcher vectors are centred (step 3), so centre the query the same way
        q = q_raw - np.load(PROC / "researcher_mean.npy")
        q /= np.linalg.norm(q)

    if a.neo4j:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as drv:
            recs, _, _ = drv.execute_query("""
                CALL db.index.vector.queryNodes('researcher_embedding', $k, $q) YIELD node, score
                OPTIONAL MATCH (node)-[:AFFILIATED_WITH {primary: true}]->(i)
                RETURN node.name AS name, i.name AS inst, node.community_label AS area, score""",
                k=a.k + 1, q=q.tolist())
            print("\nResearchers (Neo4j vector index):")
            for r in recs:
                if a.like and r["name"] == me["name"]:
                    continue
                print(f"  {r['score']:.3f}  {r['name']:28s} {r['inst'] or '':45.45s} [{r['area']}]")
            if a.query:
                recs, _, _ = drv.execute_query("""
                    CALL db.index.vector.queryNodes('paper_embedding', $k, $q) YIELD node, score
                    MATCH (r:Researcher)-[:AUTHORED]->(node)
                    RETURN node.title AS title, node.year AS year, score, collect(r.name) AS authors
                    ORDER BY score DESC""", k=a.k, q=q_raw.tolist())
                print("\nPapers:")
                for r in recs:
                    print(f"  {r['score']:.3f}  {r['title'][:80]} ({r['year']}) - {', '.join(r['authors'])}")
        return

    sims = E @ q
    order = [i for i in np.argsort(-sims) if not (a.like and r_ids[i] == me["id"])][: a.k]
    print("\nResearchers:")
    for i in order:
        r = R[r_ids[i]]
        print(f"  {sims[i]:.3f}  {r['name']:28s} {(r['institution'] or {}).get('name') or '':45.45s} [{r['community_label']}]")
    if a.query:
        P = np.load(PROC / "paper_embeddings.npy").astype(np.float32)
        p_ids = pd.read_parquet(PROC / "paper_ids.parquet").paper_id.tolist()
        papers = pd.read_parquet(PROC / "papers.parquet").set_index("paper_id")
        auth = pd.read_parquet(PROC / "authorships.parquet")
        ps = P @ q_raw
        print("\nPapers:")
        for i in np.argsort(-ps)[: a.k]:
            pid = p_ids[i]
            names = [R[x]["name"] for x in auth.loc[auth.paper_id == pid, "researcher_id"] if x in R]
            print(f"  {ps[i]:.3f}  {papers.at[pid, 'title'][:80]} ({papers.at[pid, 'year']}) - {', '.join(names)}")


if __name__ == "__main__":
    main()
