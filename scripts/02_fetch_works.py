"""Step 2 - fetch each researcher's most-cited works and their co-author counts.

Per researcher (2 requests):
  * /works sorted by citations (top WORKS_PER_RESEARCHER) with abstracts, topics, authorships
  * /works group_by=authorships.author.id -> joint-work counts over ALL their works

Outputs (data/processed/):
  papers.parquet       one row per unique paper (versions of the same paper merged)
  authorships.parquet  researcher_id, paper_id, in_top_list, author_position
  coauthors.parquet    a, b, joint_works   (both researchers in the selected set)

Usage: python scripts/02_fetch_works.py [--limit N] [--workers 6]
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import PROC, WORKS_PER_RESEARCHER, short_id  # noqa: E402
from agraph.openalex import BudgetExhausted, OpenAlex, reconstruct_abstract  # noqa: E402

WORK_TYPES = "article|conference-paper|preprint|book-chapter|book|review|dissertation|report|data-paper|software-paper"
WORK_SELECT = ("id,doi,title,publication_year,cited_by_count,type,"
               "abstract_inverted_index,authorships,primary_topic")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def fetch_one(oa: OpenAlex, r: dict):
    ids = "|".join(r["openalex_ids"])
    works = oa.get("/works", {
        "filter": f"author.id:{ids},type:{WORK_TYPES}",
        "sort": "cited_by_count:desc", "per-page": WORKS_PER_RESEARCHER, "select": WORK_SELECT,
    })["results"]
    groups = oa.get("/works", {
        "filter": f"author.id:{ids}", "group_by": "authorships.author.id", "per-page": 200,
    }).get("group_by", [])
    return r["id"], works, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N researchers (testing)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    researchers = json.loads((PROC / "researchers.json").read_text())
    if args.limit:
        researchers = researchers[: args.limit]
    # any OpenAlex id (incl. merged split profiles) -> canonical researcher id
    canon = {oid: r["id"] for r in researchers for oid in r["openalex_ids"]}

    oa = OpenAlex()
    results = {}
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(fetch_one, oa, r) for r in researchers]
            for f in tqdm(as_completed(futs), total=len(futs), desc="researchers"):
                rid, works, groups = f.result()
                results[rid] = (works, groups)
    except BudgetExhausted as e:
        print(f"\n{e}\nFetched {len(results)}/{len(researchers)} researchers before stopping.")
        print(oa.stats())
        sys.exit(2)

    papers: dict[str, dict] = {}        # canonical paper id -> row
    dedupe: dict[str, str] = {}         # title+first-author key -> canonical paper id
    alias: dict[str, str] = {}          # any work id -> canonical paper id
    auth: dict[tuple, dict] = {}        # (researcher, paper) -> row
    coauth: dict[tuple, int] = {}

    for r in researchers:
        works, groups = results[r["id"]]
        for w in works:
            wid = short_id(w["id"])
            title = (w.get("title") or "").strip()
            if not title:
                continue
            authors = w.get("authorships") or []
            first = norm((authors[0]["author"].get("display_name") or "").split(" ")[-1]) if authors else ""
            key = norm(title)
            key = f"{key}|{first}" if len(key) >= 15 else wid
            pid = dedupe.setdefault(key, wid)
            alias[wid] = pid
            pt = w.get("primary_topic") or {}
            row = {
                "paper_id": pid, "title": title,
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                "year": w.get("publication_year"), "cited_by_count": w.get("cited_by_count") or 0,
                "doi": w.get("doi"), "type": w.get("type"), "n_authors": len(authors),
                "topic_id": short_id(pt.get("id")), "topic": pt.get("display_name"),
                "subfield_id": short_id((pt.get("subfield") or {}).get("id")),
                "subfield": (pt.get("subfield") or {}).get("display_name"),
                "field_id": short_id((pt.get("field") or {}).get("id")),
                "field": (pt.get("field") or {}).get("display_name"),
                "domain_id": short_id((pt.get("domain") or {}).get("id")),
                "domain": (pt.get("domain") or {}).get("display_name"),
            }
            old = papers.get(pid)
            if old is None:
                papers[pid] = row
            else:  # another version of the same paper: keep the richer record
                if not old["abstract"] and row["abstract"]:
                    old["abstract"] = row["abstract"]
                old["cited_by_count"] = max(old["cited_by_count"], row["cited_by_count"])
            # every selected researcher on the author list gets an authorship edge
            for pos, a in enumerate(authors):
                cid = canon.get(short_id((a.get("author") or {}).get("id")))
                if cid is None:
                    continue
                k = (cid, pid)
                e = auth.setdefault(k, {"researcher_id": cid, "paper_id": pid,
                                        "in_top_list": False, "author_position": pos + 1})
                if cid == r["id"]:
                    e["in_top_list"] = True
            # OpenAlex lists return at most 100 authors; the query already guarantees that
            # this researcher wrote the work, so keep it even when they are beyond position 100
            own = auth.setdefault((r["id"], pid), {"researcher_id": r["id"], "paper_id": pid,
                                                   "in_top_list": True, "author_position": None})
            own["in_top_list"] = True
        for g in groups:
            other = canon.get(short_id(g["key"]))
            if other is None or other == r["id"]:
                continue
            a, b = sorted((r["id"], other))
            coauth[(a, b)] = max(coauth.get((a, b), 0), g["count"])

    pdf = pd.DataFrame(papers.values())
    adf = pd.DataFrame(auth.values())
    cdf = pd.DataFrame([{"a": a, "b": b, "joint_works": n} for (a, b), n in coauth.items()])
    pdf.to_parquet(PROC / "papers.parquet", index=False)
    (PROC / "meta.json").write_text(json.dumps({"retrieved": oa.data_date()}))
    adf.to_parquet(PROC / "authorships.parquet", index=False)
    cdf.to_parquet(PROC / "coauthors.parquet", index=False)

    print(f"papers: {len(pdf):,} unique ({(pdf.abstract.str.len() > 0).mean():.0%} with abstract)")
    print(f"authorship edges: {len(adf):,} (top-list: {adf.in_top_list.sum():,})")
    print(f"co-author pairs inside the set: {len(cdf):,}")
    print(oa.stats())


if __name__ == "__main__":
    main()
