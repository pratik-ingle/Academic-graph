"""Step 5 - load the knowledge graph into Neo4j (with vector indexes).

Schema
  (:Researcher)-[:AFFILIATED_WITH {years, primary}]->(:Institution)
  (:Researcher)-[:WORKS_ON {count}]->(:Topic)
  (:Researcher)-[:AUTHORED {position, top_work}]->(:Paper)-[:HAS_TOPIC]->(:Topic)
  (:Topic)-[:IN_SUBFIELD]->(:Subfield)-[:IN_FIELD]->(:Field)-[:IN_DOMAIN]->(:Domain)
  (:Researcher)-[:COAUTHOR {joint_works}]->(:Researcher)
  (:Researcher)-[:SIMILAR_TO {score, rank}]->(:Researcher)     # k nearest neighbours
  (:Researcher)-[:IN_COMMUNITY]->(:Community)-[:PART_OF]->(:Area)
Vector indexes (cosine, 768-d SPECTER2): researcher_embedding, paper_embedding

Usage: python scripts/05_load_neo4j.py [--keep]   (default wipes the database first)
"""
import argparse
import json
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, PROC  # noqa: E402

DIM = 768


def batched(rows, n=1000):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def run(session, query, rows, desc=None, size=1000):
    it = list(batched(rows, size))
    for b in tqdm(it, desc=desc, leave=False) if desc and len(it) > 3 else it:
        session.run(query, rows=b).consume()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="don't wipe the database first")
    args = ap.parse_args()

    R = json.loads((PROC / "researchers_enriched.json").read_text())
    comms = json.loads((PROC / "communities.json").read_text())
    areas = json.loads((PROC / "areas.json").read_text())
    papers = pd.read_parquet(PROC / "papers.parquet")
    auth = pd.read_parquet(PROC / "authorships.parquet")
    co = pd.read_parquet(PROC / "coauthors.parquet")
    r_ids = pd.read_parquet(PROC / "researcher_ids.parquet").researcher_id.tolist()
    r_emb = np.load(PROC / "researcher_embeddings.npy")
    p_ids = pd.read_parquet(PROC / "paper_ids.parquet").paper_id.tolist()
    p_emb = np.load(PROC / "paper_embeddings.npy").astype(np.float32)
    keep = set(r_ids)

    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    drv.verify_connectivity()
    with drv.session() as s:
        if not args.keep:
            print("wiping database ...")
            s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
            for rec in s.run("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name"):
                s.run(f"DROP INDEX {rec['name']} IF EXISTS").consume()

        for label in ("Researcher", "Institution", "Topic", "Subfield", "Field", "Domain", "Paper", "Community", "Area"):
            s.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()
        s.run("CREATE INDEX researcher_name IF NOT EXISTS FOR (n:Researcher) ON (n.name)").consume()
        s.run("CREATE FULLTEXT INDEX paper_title IF NOT EXISTS FOR (n:Paper) ON EACH [n.title, n.abstract]").consume()

        # ---- topic hierarchy -----------------------------------------------------------
        topics = {}
        for r in R:
            for t in r["topics"]:
                topics[t["id"]] = t
        for p in papers.dropna(subset=["topic_id"]).itertuples():
            topics.setdefault(p.topic_id, {"id": p.topic_id, "name": p.topic, "subfield_id": p.subfield_id,
                                           "subfield": p.subfield, "field_id": p.field_id, "field": p.field,
                                           "domain_id": p.domain_id, "domain": p.domain})
        run(s, """
            UNWIND $rows AS t
            MERGE (d:Domain {id: t.domain_id}) SET d.name = t.domain
            MERGE (f:Field {id: t.field_id}) SET f.name = t.field
            MERGE (sf:Subfield {id: t.subfield_id}) SET sf.name = t.subfield
            MERGE (tp:Topic {id: t.id}) SET tp.name = t.name
            MERGE (tp)-[:IN_SUBFIELD]->(sf) MERGE (sf)-[:IN_FIELD]->(f) MERGE (f)-[:IN_DOMAIN]->(d)
        """, list(topics.values()))

        # ---- areas and communities -------------------------------------------------------
        run(s, """
            UNWIND $rows AS a
            MERGE (n:Area {id: a.id})
            SET n.label = a.label, n.size = a.size, n.distinctive_topics = a.distinctive_topics
        """, areas)
        run(s, """
            UNWIND $rows AS c
            MERGE (n:Community {id: c.id})
            SET n.label = c.label, n.size = c.size, n.distinctive_topics = c.distinctive_topics,
                n.top_topics = c.top_topics
            WITH n, c MATCH (a:Area {id: c.area}) MERGE (n)-[:PART_OF]->(a)
        """, comms)

        # ---- researchers ----------------------------------------------------------------
        rows = []
        for r in R:
            cby = r.get("counts_by_year") or []
            rows.append({
                "id": r["id"], "name": r["name"], "openalex_ids": r["openalex_ids"], "orcid": r["orcid"],
                "rank": r["rank"], "cited_by_count": r["cited_by_count"], "works_count": r["works_count"],
                "h_index": r["h_index"], "i10_index": r["i10_index"], "mean_citedness_2yr": r["mean_citedness_2yr"],
                "cs_share": r["cs_share"], "robotics_share": r["robotics_share"], "is_robotics": r["is_robotics"],
                "x": r["x"], "y": r["y"], "community": r["community"],
                "area": r["area"], "area_label": r["area_label"],
                "community_label": r["community_label"], "paper_topics": r["paper_topics"],
                "citation_years": [c["year"] for c in cby], "citations_per_year": [c["cited_by_count"] for c in cby],
                "openalex_url": f"https://openalex.org/{r['id']}",
                "scholar_search_url": "https://scholar.google.com/citations?view_op=search_authors&mauthors="
                                      + urllib.parse.quote(r["name"]),
                "inst": r["institution"],
                "affiliations": [a for a in r["affiliations"] if a.get("id")],
                "topics": r["topics"],
            })
        run(s, """
            UNWIND $rows AS r
            MERGE (n:Researcher {id: r.id})
            SET n += apoc.map.removeKeys(r, ['inst', 'affiliations', 'topics'])
            WITH n, r
            MATCH (c:Community {id: r.community}) MERGE (n)-[:IN_COMMUNITY]->(c)
            WITH n, r
            UNWIND r.affiliations AS a
              MERGE (i:Institution {id: a.id})
              SET i.name = a.name, i.country = a.country, i.type = a.type
              MERGE (n)-[e:AFFILIATED_WITH]->(i)
              SET e.years = a.years, e.primary = (a.id = r.inst.id)
        """, rows, size=200)
        run(s, """
            UNWIND $rows AS r
            MATCH (n:Researcher {id: r.id})
            UNWIND r.topics AS t
              MATCH (tp:Topic {id: t.id})
              MERGE (n)-[e:WORKS_ON]->(tp) SET e.count = t.count
        """, rows, size=200)

        # ---- papers ------------------------------------------------------------------------
        prow = [{k: (None if pd.isna(v) else v) for k, v in rec.items()}
                for rec in papers[["paper_id", "title", "abstract", "year", "cited_by_count", "doi",
                                   "type", "n_authors", "topic_id"]].to_dict("records")]
        for p in prow:
            p["year"] = int(p["year"]) if p["year"] is not None else None
            p["cited_by_count"] = int(p["cited_by_count"])
            p["n_authors"] = int(p["n_authors"])
        run(s, """
            UNWIND $rows AS p
            MERGE (n:Paper {id: p.paper_id})
            SET n.title = p.title, n.abstract = p.abstract, n.year = p.year,
                n.cited_by_count = p.cited_by_count, n.doi = p.doi, n.type = p.type,
                n.n_authors = p.n_authors, n.openalex_url = 'https://openalex.org/' + p.paper_id
            WITH n, p WHERE p.topic_id IS NOT NULL
            MATCH (t:Topic {id: p.topic_id}) MERGE (n)-[:HAS_TOPIC]->(t)
        """, prow, desc="papers")

        arow = auth[auth.researcher_id.isin(keep)].to_dict("records")
        for a in arow:
            a["author_position"] = int(a["author_position"]) if pd.notna(a["author_position"]) else None
            a["in_top_list"] = bool(a["in_top_list"])
        run(s, """
            UNWIND $rows AS a
            MATCH (r:Researcher {id: a.researcher_id}), (p:Paper {id: a.paper_id})
            MERGE (r)-[e:AUTHORED]->(p) SET e.position = a.author_position, e.top_work = a.in_top_list
        """, arow, desc="authorships", size=5000)

        # ---- researcher <-> researcher -----------------------------------------------------------
        crow = [{"a": a, "b": b, "w": int(w)} for a, b, w in co.itertuples(index=False) if a in keep and b in keep]
        run(s, """
            UNWIND $rows AS c
            MATCH (a:Researcher {id: c.a}), (b:Researcher {id: c.b})
            MERGE (a)-[e:COAUTHOR]->(b) SET e.joint_works = c.w
        """, crow, size=5000)
        srow = [{"a": r["id"], "b": x["id"], "score": x["score"], "rank": i + 1}
                for r in R for i, x in enumerate(r["similar"])]
        run(s, """
            UNWIND $rows AS e
            MATCH (a:Researcher {id: e.a}), (b:Researcher {id: e.b})
            MERGE (a)-[s:SIMILAR_TO]->(b) SET s.score = e.score, s.rank = e.rank
        """, srow, size=5000)

        # ---- embeddings + vector indexes (stored as float32 vectors) -------------------------
        run(s, """
            UNWIND $rows AS r MATCH (n:Researcher {id: r.id})
            CALL db.create.setNodeVectorProperty(n, 'embedding', r.v)
        """, [{"id": i, "v": v.tolist()} for i, v in zip(r_ids, r_emb)], size=500)
        run(s, """
            UNWIND $rows AS r MATCH (n:Paper {id: r.id})
            CALL db.create.setNodeVectorProperty(n, 'embedding', r.v)
        """, [{"id": i, "v": v.tolist()} for i, v in zip(p_ids, p_emb)], desc="paper vectors", size=1000)
        for name, label in (("researcher_embedding", "Researcher"), ("paper_embedding", "Paper")):
            s.run(f"""CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.embedding)
                      OPTIONS {{indexConfig: {{`vector.dimensions`: {DIM},
                                               `vector.similarity_function`: 'cosine'}}}}""").consume()
        s.run("CALL db.awaitIndexes(600)").consume()

        print("loaded:")
        for rec in s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"):
            print(f"  {rec['label']:12s} {rec['n']:>8,}")
        for rec in s.run("MATCH ()-[e]->() RETURN type(e) AS type, count(*) AS n ORDER BY n DESC"):
            print(f"  {rec['type']:16s} {rec['n']:>8,}")
    drv.close()


if __name__ == "__main__":
    main()
