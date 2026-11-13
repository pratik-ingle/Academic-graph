"""Step 6 - build the interactive atlas page from the processed graph.

Writes
  web/index.html   standalone page (open it in a browser, no server needed)
  web/atlas.html   same page without the <html>/<head>/<body> wrapper (for publishing)
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import KNN_K, PROC, SEED, WEB, WORKS_PER_RESEARCHER  # noqa: E402
from analysis.common import fingerprint  # noqa: E402

K_SHOWN = 5  # similarity lines drawn per researcher (the panel lists all KNN_K)


SHORT = {"13": "Biology", "28": "Neuroscience", "27": "Medicine", "16": "Chemistry", "25": "Materials science",
         "31": "Physics & astronomy", "26": "Mathematics", "19": "Earth sciences", "20": "Economics",
         "32": "Psychology", "33": "Social sciences", "12": "Arts & humanities", "17": "Computer science"}


def _clean(x):
    """NaN -> None so the page data stays valid JSON."""
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def beyond_data(path):
    """Cross-field results from analysis/fields_analyze.py for the Beyond CS view."""
    res = json.loads(path.read_text())
    popf = path.parent / "population.json"
    pop_json = json.loads(popf.read_text()) if popf.exists() else {}
    fields = []
    for fid, name in res["fields"].items():
        r = res["researchers"].get(fid)
        if not r:
            continue
        top, popn = res["top"][fid], res["population"][fid]
        top_oa = pop_json.get("top_oa", {}).get(fid)
        fields.append({
            "id": fid, "name": name, "short": SHORT.get(fid, name),
            "res": {k: r.get(k) for k in ("n", "then", "now", "delta", "ci", "p", "any_now", "llm_now", "strict")},
            "top": {y: v for y, v in top.items() if y.isdigit()}, "about_ai": top.get("about_ai"),
            "pop": {y: v for y, v in popn.items() if y.isdigit()}, "premium": popn.get("premium"),
            "examples": res["examples"].get(fid, []),
            "top_oa": top_oa,
        })
    years = sorted({int(y) for f in fields for y in f["pop"]})
    out = {"fields": fields, "years": years, "partial_year": pop_json.get("partial_year"),
           "retrieved": pop_json.get("retrieved")}
    chk = path.parent / "check_2026.json"   # title-only vs title+abstract check of the partial year
    if chk.exists():
        out["check2026"] = json.loads(chk.read_text())
    return _clean(out)


def layout_3d(R):
    """3-D UMAP of the researcher profiles for the page's 3D view, same settings as the 2-D map
    (scripts/04_build_graph.py) but three components. Rotated (orthogonal Procrustes) so its
    first two axes line up with the 2-D map, then scaled to unit RMS radius. Cached in
    data/processed/layout_3d.npz, keyed on the embeddings and researcher order."""
    import hashlib
    ids = pd.read_parquet(PROC / "researcher_ids.parquet").researcher_id.tolist()
    emb_f = PROC / "researcher_embeddings.npy"
    key = hashlib.sha1(emb_f.read_bytes() + "|".join(ids).encode() + f"|umap3|{SEED}".encode()).hexdigest()[:16]
    cache = PROC / "layout_3d.npz"
    P = None
    if cache.exists():
        z = np.load(cache)
        if str(z["key"]) == key:
            P = z["P"]
    if P is None:
        E = np.load(emb_f)
        n = len(ids)
        if n >= 10:
            import umap
            P = umap.UMAP(n_components=3, n_neighbors=min(15, n - 1), min_dist=0.12, metric="cosine",
                          random_state=SEED).fit_transform(E)
        else:
            P = np.random.default_rng(SEED).normal(size=(n, 3))
        np.savez(cache, key=np.array(key), P=P)
    row = {rid: i for i, rid in enumerate(ids)}
    P = P[[row[r["id"]] for r in R]].astype(float)
    P -= P.mean(0)
    XY = np.array([[r["x"], r["y"], 0.0] for r in R]); XY -= XY.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ XY)
    P = P @ (U @ Vt)
    return P / np.sqrt((P ** 2).sum(1).mean())


def main():
    R = json.loads((PROC / "researchers_enriched.json").read_text())
    comms = json.loads((PROC / "communities.json").read_text())
    areas = json.loads((PROC / "areas.json").read_text())
    co = pd.read_parquet(PROC / "coauthors.parquet")
    n_papers = len(pd.read_parquet(PROC / "paper_ids.parquet"))
    idx = {r["id"]: i for i, r in enumerate(R)}
    meta_f = PROC / "meta.json"
    retrieved = json.loads(meta_f.read_text())["retrieved"] if meta_f.exists() else dt.date.today().isoformat()
    today = dt.date.fromisoformat(retrieved)   # dates on the page follow the data, not the build

    P3 = layout_3d(R)
    people = []
    for r, p3 in zip(R, P3):
        inst = r["institution"] or {}
        people.append({
            "id": r["id"], "name": r["name"], "inst": inst.get("name"), "cc": inst.get("country"),
            "cites": r["cited_by_count"], "h": r["h_index"], "works": r["works_count"], "rank": r["rank"],
            "x": r["x"], "y": r["y"], "p3": [round(float(v), 3) for v in p3], "c": r["community"], "a": r["area"],
            "rob": r["robotics_share"], "orcid": r["orcid"],
            "topics": [t["name"] for t in r["topics"]],
            "papers": [[p["title"], p["year"], p["cited_by_count"], p["doi"] if isinstance(p["doi"], str) else f"https://openalex.org/{p['id']}"]
                       for p in r["top_papers"]],
            "sim": [[idx[s["id"]], round(s["score"], 3)] for s in r["similar"] if s["id"] in idx],
            "cby": [[c["year"], c["cited_by_count"]] for c in r["counts_by_year"]],
        })

    coauthor = [[idx[a], idx[b], int(w)] for a, b, w in co.itertuples(index=False) if a in idx and b in idx]
    sim_edges = {}
    for i, p in enumerate(people):
        for j, s in p["sim"][:K_SHOWN]:
            sim_edges[tuple(sorted((i, j)))] = s
    similar = [[i, j, s] for (i, j), s in sim_edges.items()]

    xs = np.array([p["x"] for p in people]); ys = np.array([p["y"] for p in people])
    cs = np.array([p["c"] for p in people])
    med3 = lambda m: [round(float(v), 3) for v in np.median(P3[m], axis=0)]  # noqa: E731
    communities = [{"id": c["id"], "label": c["label"], "area": c["area"], "size": c["size"],
                    "x": round(float(np.median(xs[cs == c["id"]])), 4),
                    "y": round(float(np.median(ys[cs == c["id"]])), 4), "p3": med3(cs == c["id"])} for c in comms]
    av = np.array([p["a"] for p in people])
    area_list = [{"id": a["id"], "label": a["label"], "size": a["size"],
                  "x": round(float(np.median(xs[av == a["id"]])), 4),
                  "y": round(float(np.median(ys[av == a["id"]])), 4), "p3": med3(av == a["id"]),
                  "top": [idx[m] for m in a["top_members"] if m in idx]} for a in areas]

    data = {
        "meta": {
            "generated": dt.date.today().isoformat(), "retrieved": retrieved, "last_full_year": today.year - 1,
            "n": len(people), "n_papers": n_papers, "n_coauthor": len(coauthor),
            "n_communities": len(communities), "works_per": WORKS_PER_RESEARCHER,
            "k": KNN_K, "k_shown": K_SHOWN,
            "min_cites": min(p["cites"] for p in people), "max_cites": max(p["cites"] for p in people),
        },
        "areas": area_list, "communities": communities, "researchers": people,
        "coauthor": coauthor, "similar": similar,
    }
    fpath = PROC / "findings.json"
    if fpath.exists():
        findings = json.loads(fpath.read_text())
        same_people = findings.pop("ids", None) == [p["id"] for p in people]
        if same_people and findings.get("fingerprint") == fingerprint(PROC):
            data["findings"] = findings
        else:
            print("findings.json is stale (graph changed); rerun analysis/export_findings.py")
    fres = PROC.parent / "fields" / "results.json"
    if fres.exists():
        data["beyond"] = beyond_data(fres)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    tpl = (WEB / "template.html").read_text()
    page = tpl.replace("/*__DATA__*/", blob)
    head, body = page.split("<!--BODY-->", 1)
    (WEB / "atlas.html").write_text(head + body)
    (WEB / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + "</head>\n<body>\n" + body + "</body>\n</html>\n")
    size = (WEB / "index.html").stat().st_size
    print(f"web/index.html ({size / 1e6:.2f} MB): {len(people)} researchers, "
          f"{len(coauthor)} co-author + {len(similar)} similarity lines, "
          f"{len(area_list)} areas / {len(communities)} communities")


if __name__ == "__main__":
    main()
