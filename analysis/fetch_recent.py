"""Fetch each researcher's most-cited works published since 2022 and embed them with
SPECTER2, so their recent research can be compared with the work they are known for.

Outputs (data/processed/): recent_papers.parquet, recent_authorships.parquet,
recent_embeddings.npy (float16, row order = recent_papers.parquet)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import PROC, short_id  # noqa: E402
from agraph.openalex import OpenAlex, reconstruct_abstract  # noqa: E402

TYPES = "article|conference-paper|preprint|book-chapter|review"
SINCE = "2022-01-01"
PER = 30


def main():
    R = json.loads((PROC / "researchers.json").read_text())
    oa = OpenAlex()

    def one(r):
        d = oa.get("/works", {
            "filter": f"author.id:{'|'.join(r['openalex_ids'])},type:{TYPES},from_publication_date:{SINCE}",
            "sort": "cited_by_count:desc", "per-page": PER,
            "select": "id,title,publication_year,cited_by_count,abstract_inverted_index,primary_topic"})
        return r["id"], d["results"]

    papers, auth = {}, []
    with ThreadPoolExecutor(6) as ex:
        for rid, works in tqdm(ex.map(one, R), total=len(R), desc="recent works"):
            for w in works:
                if not w.get("title"):
                    continue
                pid = short_id(w["id"])
                pt = w.get("primary_topic") or {}
                papers[pid] = {"paper_id": pid, "title": w["title"], "year": w["publication_year"],
                               "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                               "cited_by_count": w.get("cited_by_count") or 0, "topic": pt.get("display_name"),
                               "subfield": (pt.get("subfield") or {}).get("display_name")}
                auth.append({"researcher_id": rid, "paper_id": pid})
    P = pd.DataFrame(papers.values())
    A = pd.DataFrame(auth).drop_duplicates()
    # versions of one paper (preprint + venue) count once per researcher: keep the most-cited
    P["key"] = P.title.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    A = A.join(P.set_index("paper_id")[["key", "cited_by_count"]], on="paper_id")
    A = A.sort_values("cited_by_count", ascending=False).drop_duplicates(["researcher_id", "key"])[["researcher_id", "paper_id"]]
    P = P[P.paper_id.isin(A.paper_id)].drop(columns="key").reset_index(drop=True)
    from agraph.embed import Specter2
    enc = Specter2("doc")
    V = enc.encode([enc.paper_text(t, a) for t, a in zip(P.title, P.abstract)], progress=True)
    P.to_parquet(PROC / "recent_papers.parquet", index=False)
    A.to_parquet(PROC / "recent_authorships.parquet", index=False)
    np.save(PROC / "recent_embeddings.npy", V.astype(np.float16))
    print(f"{len(P):,} recent papers for {A.researcher_id.nunique()} researchers; {oa.stats()}")


if __name__ == "__main__":
    main()
