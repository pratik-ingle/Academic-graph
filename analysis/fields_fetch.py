"""Cross-field data: is research in other fields moving toward AI?

For each field (OpenAlex field ids below):
  1. top researchers: the N most-cited authors with >= 50% of their topics in the field;
     their 25 most-cited papers up to 2016 ("then") and since 2022 ("now"),
     papers with > 50 authors excluded (mega-collaborations say little about a person's focus)
  2. top-cited papers: the 100 most-cited papers per publication year, 2014-2025
  3. whole field: number of papers per year, and how many match the AI / LLM search

Outputs: data/fields/*.parquet, data/fields/population.json
Cost: ~2,600 normal requests + ~240 full-text searches (~$0.5 of the daily OpenAlex budget).
"""
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import DATA, short_id  # noqa: E402
from agraph.openalex import OpenAlex, reconstruct_abstract  # noqa: E402

OUT = DATA / "fields"
OUT.mkdir(exist_ok=True)
FIELDS = {
    "13": "Biology (biochemistry, genetics, molecular)", "28": "Neuroscience", "27": "Medicine",
    "16": "Chemistry", "25": "Materials Science", "31": "Physics & Astronomy", "26": "Mathematics",
    "19": "Earth & Planetary Sciences", "20": "Economics & Finance", "32": "Psychology",
    "33": "Social Sciences", "12": "Arts & Humanities", "17": "Computer Science",
}
N_PER_FIELD = 100
TYPES = "article|review|conference-paper|preprint|book-chapter"
YEARS = range(2014, dt.date.today().year + 1)   # the current year is partial (to date)
LLM_YEARS = range(2020, dt.date.today().year + 1)
AI_QUERY = ('"machine learning" OR "deep learning" OR "artificial intelligence" OR "artificial neural network" '
            'OR "convolutional neural" OR "large language model" OR "random forest" OR "support vector machine" '
            'OR "graph neural network" OR "ChatGPT" OR "generative AI"')
LLM_QUERY = '"large language model" OR "ChatGPT" OR "GPT-4" OR "generative AI"'
PREMIUM_YEARS = (2022, 2023, 2024)
SELECT = "id,title,publication_year,cited_by_count,abstract_inverted_index,primary_topic"


def paper_rows(works, **extra):
    rows = []
    for w in works:
        if not w.get("title"):
            continue
        pt = w.get("primary_topic") or {}
        rows.append({"paper_id": short_id(w["id"]), "title": w["title"], "year": w.get("publication_year"),
                     "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                     "cited_by_count": w.get("cited_by_count") or 0, "topic": pt.get("display_name"),
                     "subfield": (pt.get("subfield") or {}).get("display_name"),
                     "field_id": short_id((pt.get("field") or {}).get("id")), **extra})
    return rows


def select_researchers(oa, fid, taken):
    """taken: researcher ids already assigned to an earlier field (each person counts once)."""
    picked, page = [], 0
    while len(picked) < N_PER_FIELD and page < 10:
        page += 1
        res = oa.get("/authors", {"filter": f"topics.field.id:{fid}", "sort": "cited_by_count:desc",
                                  "per-page": 200, "page": page,
                                  "select": "id,display_name,works_count,cited_by_count,topics,last_known_institutions"})["results"]
        for a in res:
            tops = a.get("topics") or []
            tot = sum(t["count"] for t in tops) or 1
            share = sum(t["count"] for t in tops if short_id(t["field"]["id"]) == fid) / tot
            if share >= 0.5 and a.get("works_count", 0) >= 5 and len(a["display_name"].split()) >= 2 \
                    and "team" not in a["display_name"].lower() and short_id(a["id"]) not in taken:
                taken.add(short_id(a["id"]))
                inst = (a.get("last_known_institutions") or [{}])[0] or {}
                picked.append({"researcher_id": short_id(a["id"]), "name": a["display_name"], "field_id": fid,
                               "cited_by_count": a["cited_by_count"], "works_count": a["works_count"],
                               "field_share": round(share, 3), "institution": inst.get("display_name")})
            if len(picked) == N_PER_FIELD:
                break
        if not res:
            break
    return picked


def main():
    oa = OpenAlex()
    # 1. researchers -------------------------------------------------------------------
    researchers, taken = [], set()
    for fid in tqdm(FIELDS, desc="selecting researchers"):
        researchers += select_researchers(oa, fid, taken)
    pd.DataFrame(researchers).to_parquet(OUT / "researchers.parquet", index=False)

    def works_of(r):
        out = []
        for period, yf in (("then", "publication_year:<2017"), ("now", "publication_year:>2021")):
            res = oa.get("/works", {"filter": f"author.id:{r['researcher_id']},{yf},authors_count:<51,type:{TYPES}",
                                    "sort": "cited_by_count:desc", "per-page": 25, "select": SELECT})["results"]
            out += paper_rows(res, researcher_id=r["researcher_id"], period=period)
        return out

    rows = []
    with ThreadPoolExecutor(6) as ex:
        for part in tqdm(ex.map(works_of, researchers), total=len(researchers), desc="researcher papers"):
            rows += part
    pd.DataFrame(rows).to_parquet(OUT / "researcher_papers.parquet", index=False)

    # 2. top-cited papers per field-year ----------------------------------------------------
    def top_of(fy):
        fid, y = fy
        res = oa.get("/works", {"filter": f"primary_topic.field.id:{fid},publication_year:{y},type:{TYPES}",
                                "sort": "cited_by_count:desc", "per-page": 100, "select": SELECT})["results"]
        return paper_rows(res, field=fid, pub_year=y)

    rows = []
    with ThreadPoolExecutor(6) as ex:
        for part in tqdm(ex.map(top_of, [(f, y) for f in FIELDS for y in YEARS]), total=len(FIELDS) * len(YEARS), desc="top-cited papers"):
            rows += part
    top_df = pd.DataFrame(rows)
    top_df.to_parquet(OUT / "top_papers.parquet", index=False)

    # The same top-cited set measured with OpenAlex's own search (the detector used for the
    # whole-field counts), so the citation premium compares like with like: all papers of the
    # field-year cited at least as often as its 100th most-cited paper.
    top_oa = {}
    for f in FIELDS:
        for y in PREMIUM_YEARS:
            g = top_df[(top_df.field == f) & (top_df.pub_year == y)]
            if g.empty:
                continue
            thr = int(g.cited_by_count.min())
            base = f"primary_topic.field.id:{f},publication_year:{y},type:{TYPES},cited_by_count:>{thr - 1}"
            n_all = oa.get("/works", {"filter": base, "per-page": 1, "select": "id"})["meta"]["count"]
            n_ai = oa.get("/works", {"filter": base + f",title_and_abstract.search:{AI_QUERY}", "per-page": 1, "select": "id"})["meta"]["count"]
            top_oa.setdefault(f, {})[str(y)] = {"all": n_all, "ai": n_ai, "threshold": thr}

    # 3. whole-field counts ---------------------------------------------------------------------
    def count(fid, y, query=None):
        flt = f"primary_topic.field.id:{fid},publication_year:{y},type:{TYPES}"
        if query:
            flt += f",title_and_abstract.search:{query}"
        return oa.get("/works", {"filter": flt, "per-page": 1, "select": "id"})["meta"]["count"]

    jobs = [(f, y, None) for f in FIELDS for y in YEARS] + [(f, y, AI_QUERY) for f in FIELDS for y in YEARS] + \
           [(f, y, LLM_QUERY) for f in FIELDS for y in LLM_YEARS]
    pop = {}
    # full-text searches are rate-limited more tightly: run them one at a time
    for f, y, q in tqdm(jobs, desc="field counts"):
        pop.setdefault(f, {}).setdefault(str(y), {})[{None: "all", AI_QUERY: "ai", LLM_QUERY: "llm"}[q]] = count(f, y, q)
    (OUT / "population.json").write_text(json.dumps({"fields": FIELDS, "counts": pop,
                                                     "ai_query": AI_QUERY, "llm_query": LLM_QUERY,
                                                     "retrieved": oa.data_date(), "partial_year": max(YEARS),
                                                     "top_oa": top_oa}, indent=1))
    print(f"{len(researchers)} researchers; {oa.stats()}")


if __name__ == "__main__":
    main()
