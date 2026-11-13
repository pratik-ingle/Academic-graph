"""Step 1 - pick the N most-cited researchers whose work is mainly CS or robotics.

Two candidate pools are paged from OpenAlex in descending citation order:
  * authors with any Computer Science topic
  * authors with any robotics topic (many robotics topics sit under Engineering)
A candidate qualifies when >= MIN_FIELD_SHARE of their topic counts is CS or
robotics and the profile looks like a person. Paging continues until both pools
are below the citation count of the N-th qualifier, so the ranking is exact.

Output: data/processed/researchers.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agraph.config import (AMBIGUOUS_ROBOTICS_TOPICS, CS_FIELD_ID, MIN_FIELD_SHARE,  # noqa: E402
                           MIN_WORKS, N_RESEARCHERS, PROC, ROBOTICS_LABEL_SHARE,
                           ROBOTICS_TOPICS, short_id)
from agraph.openalex import OpenAlex  # noqa: E402

SELECT = ("id,display_name,orcid,works_count,cited_by_count,summary_stats,"
          "affiliations,last_known_institutions,topics,counts_by_year")

NON_PERSON = re.compile(
    r"\b(team|consortium|collaboration|group|committee|project|organi[sz]ation|association|"
    r"network|initiative|society|working|investigators|contributors|developers|"
    r"study|program(me)?|survey|institute|university|laborator(y|ies)|inc|ltd|corp|"
    r"foundation|council|agency|department|centre|center|editors?|anonymous|unknown|"
    r"et al|community|members|authors)\b", re.I)


CUTOFF_MARGIN = 0.85


def clean_name(name: str) -> str:
    name = name.title() if name.isupper() else name
    parts = name.strip().split()
    # "Takeo Kanade A" -> "Takeo Kanade", but keep "Ed H." (only strip when 2+ words remain)
    if len(parts) >= 3 and re.fullmatch(r"[A-Z]\.?", parts[-1]):
        parts = parts[:-1]
    return " ".join(parts)


def is_person(name: str) -> bool:
    if not name or NON_PERSON.search(name):
        return False
    return len(name.split()) >= 2  # single tokens are usually acronyms/organisations


def shares(author: dict) -> tuple[float, float]:
    topics = author.get("topics") or []
    total = sum(t.get("count", 0) for t in topics)
    if not total:
        return 0.0, 0.0
    tid = lambda t: short_id(t["id"])  # noqa: E731
    robotics = set(ROBOTICS_TOPICS)
    if not any(tid(t) in robotics - AMBIGUOUS_ROBOTICS_TOPICS for t in topics):
        robotics -= AMBIGUOUS_ROBOTICS_TOPICS  # deep RL without hands-on robotics counts as CS
    rob = sum(t["count"] for t in topics if tid(t) in robotics)
    cs = sum(t["count"] for t in topics
             if short_id(t["field"]["id"]) == CS_FIELD_ID and tid(t) not in robotics)
    return cs / total, rob / total


def primary_institution(author: dict) -> dict | None:
    """Most consistent recent affiliation (last_known_institutions is often noisy)."""
    affs = author.get("affiliations") or []
    if not affs:
        lk = author.get("last_known_institutions") or []
        return lk[0] if lk else None
    latest = max((max(a["years"]) for a in affs if a.get("years")), default=0)
    window = range(latest - 4, latest + 1)

    def score(a):
        yrs = a.get("years") or []
        return (sum(y in window for y in yrs), max(yrs, default=0), len(yrs))

    return max(affs, key=score)["institution"]


def slim_inst(inst: dict | None) -> dict | None:
    if not inst:
        return None
    return {"id": short_id(inst.get("id")), "name": inst.get("display_name"),
            "country": inst.get("country_code"), "type": inst.get("type")}


def to_record(a: dict, cs: float, rob: float) -> dict:
    ss = a.get("summary_stats") or {}
    return {
        "id": short_id(a["id"]),
        "name": clean_name(a["display_name"]),
        "openalex_ids": [short_id(a["id"])],
        "orcid": a.get("orcid"),
        "works_count": a.get("works_count", 0),
        "cited_by_count": a.get("cited_by_count", 0),
        "h_index": ss.get("h_index"),
        "i10_index": ss.get("i10_index"),
        "mean_citedness_2yr": ss.get("2yr_mean_citedness"),
        "institution": slim_inst(primary_institution(a)),
        "affiliations": [
            {**slim_inst(x["institution"]), "years": sorted(x.get("years") or [])}
            for x in (a.get("affiliations") or [])
        ],
        "topics": [
            {"id": short_id(t["id"]), "name": t["display_name"], "count": t["count"],
             "subfield_id": short_id(t["subfield"]["id"]), "subfield": t["subfield"]["display_name"],
             "field_id": short_id(t["field"]["id"]), "field": t["field"]["display_name"],
             "domain_id": short_id(t["domain"]["id"]), "domain": t["domain"]["display_name"]}
            for t in (a.get("topics") or [])
        ],
        "cs_share": round(cs, 3),
        "robotics_share": round(rob, 3),
        "is_robotics": rob >= ROBOTICS_LABEL_SHARE,
        "counts_by_year": sorted(
            [{"year": c["year"], "cited_by_count": c["cited_by_count"], "works_count": c["works_count"]}
             for c in (a.get("counts_by_year") or [])], key=lambda c: c["year"]),
    }


def _same_person(a: dict, b: dict) -> bool:
    ia, ib = (a["institution"] or {}).get("id"), (b["institution"] or {}).get("id")
    ta, tb = {t["id"] for t in a["topics"]}, {t["id"] for t in b["topics"]}
    jac = len(ta & tb) / max(1, len(ta | tb))
    # Same institution alone is not enough: two different "Hang Su"s at Tsinghua share no topics.
    return (ia is not None and ia == ib and bool(ta & tb)) or jac >= 0.5


def _initials(name: str) -> tuple[str, set]:
    parts = re.sub(r"[.\-]", " ", name).split()
    return re.sub(r"[^a-z]", "", parts[-1].lower()), {p[0].lower() for p in parts[:-1] if p}


def _given_compatible(a: str, b: str) -> bool:
    """'A.Y. Ng'~'Andrew Y. Ng', 'Jon Shlens'~'Jonathon Shlens', 'Alex J. Smola'~'Alexander Johannes Smola'."""
    pa, pb = re.sub(r"[.\-]", " ", a).split()[:-1], re.sub(r"[.\-]", " ", b).split()[:-1]
    if not pa or not pb:
        return False
    fa, fb = pa[0].lower(), pb[0].lower()
    first_ok = fa == fb or (min(len(fa), len(fb)) == 1 and fa[0] == fb[0]) or \
        (min(len(fa), len(fb)) >= 3 and (fa.startswith(fb) or fb.startswith(fa)))
    ia, ib = {x[0].lower() for x in pa}, {x[0].lower() for x in pb}
    return first_ok and (ia <= ib or ib <= ia)


def merge_name_variants(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Merge profiles whose names differ only in spelling-out ('L. A. Zadeh' / 'Lotfi A. Zadeh')
    when they share a primary institution."""
    by_sur: dict[str, list[dict]] = {}
    for r in records:
        by_sur.setdefault(_initials(r["name"])[0], []).append(r)
    out, log, gone = [], [], set()
    for grp in by_sur.values():
        grp.sort(key=lambda r: -r["cited_by_count"])
        for i, head in enumerate(grp):
            if id(head) in gone:
                continue
            for other in grp[i + 1:]:
                if id(other) in gone:
                    continue
                hi, oi = (head["institution"] or {}).get("id"), (other["institution"] or {}).get("id")
                shared = {t["id"] for t in head["topics"]} & {t["id"] for t in other["topics"]}
                if hi and hi == oi and shared and _given_compatible(head["name"], other["name"]):
                    _merge_into(head, other)
                    gone.add(id(other))
                    log.append(f"{other['name']} -> {head['name']}")
            out.append(head)
    return out, log


def absorb_fragments(records: list[dict]) -> tuple[list[dict], list[str], list[tuple]]:
    """Profiles with < MIN_WORKS works are fragments (e.g. 'D.P. Kingma' holding only the
    Adam paper) or ghosts. Fold a fragment into the single full profile with the same
    surname, a shared initial and a shared topic; drop it otherwise."""
    full = [r for r in records if r["works_count"] >= MIN_WORKS]
    frags = [r for r in records if r["works_count"] < MIN_WORKS]
    log, dropped = [], []
    for f in frags:
        sur, ini = _initials(f["name"])
        ftop = {t["id"] for t in f["topics"]}
        # same surname, and either all the fragment's initials appear in the full name
        # ('D.P. Kingma' -> 'Diederik P. Kingma') or one initial plus a shared topic
        match = [r for r in full if _initials(r["name"])[0] == sur and (
                 ini <= _initials(r["name"])[1]
                 or (ini & _initials(r["name"])[1] and ftop & {t["id"] for t in r["topics"]}))]
        if len(match) == 1:
            _merge_into(match[0], f)
            log.append(f"{f['name']} ({f['works_count']} works) -> {match[0]['name']}")
        else:
            dropped.append((f["name"], f["cited_by_count"], f"only {f['works_count']} works"))
    return full, log, dropped


def _merge_into(head: dict, r: dict) -> None:
    head["openalex_ids"] = head["openalex_ids"] + r["openalex_ids"]
    head["cited_by_count"] += r["cited_by_count"]
    head["works_count"] += r["works_count"]
    head["h_index"] = max(head["h_index"] or 0, r["h_index"] or 0)
    head["i10_index"] = max(head["i10_index"] or 0, r["i10_index"] or 0)
    years = {c["year"]: dict(c) for c in head["counts_by_year"]}
    for c in r["counts_by_year"]:
        y = years.setdefault(c["year"], {"year": c["year"], "cited_by_count": 0, "works_count": 0})
        y["cited_by_count"] += c["cited_by_count"]
        y["works_count"] += c["works_count"]
    head["counts_by_year"] = sorted(years.values(), key=lambda c: c["year"])


def merge_split_profiles(records: list[dict]) -> tuple[list[dict], list[str]]:
    """OpenAlex sometimes splits one person into several profiles. Merge profiles
    with the same name that also share an institution or most of their topics."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(re.sub(r"[^a-z]", "", r["name"].lower()), []).append(r)
    out, log = [], []
    for grp in groups.values():
        grp.sort(key=lambda r: -r["cited_by_count"])
        while grp:
            head, rest = grp[0], grp[1:]
            same = [r for r in rest if _same_person(head, r)]
            grp = [r for r in rest if r not in same]
            if same:
                head = dict(head)
                for r in same:
                    _merge_into(head, r)
                log.append(f"{head['name']}: {head['openalex_ids']}")
            out.append(head)
    return out, log


def main():
    oa = OpenAlex()
    pools = {
        "cs": f"topics.field.id:{CS_FIELD_ID}",
        "robotics": "topics.id:" + "|".join(ROBOTICS_TOPICS),
    }
    seen: dict[str, dict] = {}
    rejected: list[tuple[str, int, str]] = []
    page = {k: 0 for k in pools}
    floor = {k: float("inf") for k in pools}   # lowest citation count seen per pool
    done = {k: False for k in pools}

    def cutoff() -> float:
        q = sorted((r["cited_by_count"] for r in seen.values()), reverse=True)
        return q[N_RESEARCHERS - 1] if len(q) >= N_RESEARCHERS else -1

    while not all(done.values()):
        for k, flt in pools.items():
            if done[k]:
                continue
            page[k] += 1
            d = oa.get("/authors", {"filter": flt, "sort": "cited_by_count:desc",
                                    "per-page": 200, "page": page[k], "select": SELECT})
            res = d.get("results", [])
            for a in res:
                aid = short_id(a["id"])
                if aid in seen:
                    continue
                cs, rob = shares(a)
                if not is_person(a["display_name"]):
                    rejected.append((a["display_name"], a["cited_by_count"], "not a person"))
                elif cs + rob < MIN_FIELD_SHARE:
                    rejected.append((a["display_name"], a["cited_by_count"], f"cs+robotics share {cs + rob:.2f}"))
                else:
                    seen[aid] = to_record(a, cs, rob)
            floor[k] = res[-1]["cited_by_count"] if res else -1
            c = cutoff()
            # page past the cutoff with a margin: merging split profiles and dropping ghost
            # profiles later lowers the cutoff, and those candidates must already be in the pool
            if not res or (c >= 0 and floor[k] < CUTOFF_MARGIN * c) or page[k] >= 50:
                done[k] = True
            print(f"  pool={k:8s} page={page[k]:2d} floor={floor[k]:>7} qualified={len(seen):5d} cutoff={c}")

    merged, merges = merge_split_profiles(list(seen.values()))
    merged, variants = merge_name_variants(merged)
    merges += variants
    merged, absorbed, dropped = absorb_fragments(merged)
    rejected += dropped
    for line in merges + absorbed:
        print("  merged:", line)
    print("  dropped ghost profiles:", [d[0] for d in sorted(dropped, key=lambda d: -d[1])][:12])
    ranked = sorted(merged, key=lambda r: r["cited_by_count"], reverse=True)[:N_RESEARCHERS]
    if ranked and any(floor[k] >= ranked[-1]["cited_by_count"] and floor[k] > 0 for k in pools):
        print("WARNING: a candidate pool was not paged below the final cutoff; raise CUTOFF_MARGIN")
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    out = PROC / "researchers.json"
    out.write_text(json.dumps(ranked, indent=1, ensure_ascii=False))
    (PROC / "rejected_candidates.json").write_text(json.dumps(
        sorted(rejected, key=lambda x: -x[1])[:500], indent=1, ensure_ascii=False))

    nrob = sum(r["is_robotics"] for r in ranked)
    print(f"\nSelected {len(ranked)} researchers -> {out}")
    print(f"  citations: {ranked[0]['cited_by_count']:,} .. {ranked[-1]['cited_by_count']:,}")
    print(f"  tagged robotics: {nrob}")
    print(f"  top rejected (not person): {[x[0] for x in rejected if x[2] == 'not a person'][:15]}")
    print(f"  {oa.stats()}")


if __name__ == "__main__":
    main()
