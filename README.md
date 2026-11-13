# CS & Robotics researcher knowledge graph

A knowledge graph of the **1,000 most-cited researchers in computer science and robotics**,
with one SPECTER2 embedding per researcher built from their research.

**Live atlas: <https://pratik-ingle.github.io/A-graph-Academic-graph-/>** (opens in the 3D view; flat map: [`#map`](https://pratik-ingle.github.io/A-graph-Academic-graph-/#map))

It ships as:

- **`web/index.html`**: an interactive atlas. Each researcher is a dot placed by research similarity
  (a UMAP projection of their embeddings), coloured by research area, with co-author and
  similar-work links, search, a detail panel and a sortable table view, plus a rotatable 3D view
  (a separate 3-D UMAP of the same embeddings). Open the file directly; no server needed.
  The page is generated (git-ignored); the published copy is `docs/index.html`, served by GitHub Pages.
  To update the site after a rebuild: `cp web/index.html docs/index.html`, then commit and push.
- **Neo4j**: the full graph (researchers, papers, topics, institutions, communities) with vector
  indexes on researcher and paper embeddings.
- **Files** in `data/processed/`: embeddings (`.npy`), tables (`.parquet`), `graph.graphml` for
  Gephi/Cytoscape, and `graph.json`.

## Why OpenAlex rather than Google Scholar

Google Scholar has no API, blocks automated access and forbids scraping in its terms.
[OpenAlex](https://openalex.org) is an open index covering the same people and papers. It has
abstracts, topics, institutions and co-authorship, so the data comes from there. Each researcher's
panel links to a Google Scholar name search.

## Run it

```bash
# once: put a free OpenAlex key in .env  (https://openalex.org/settings/api)
#       without a key, everyone on your IP shares ~1,000 requests/day; a full build needs ~2,000
./run_all.sh
```

Or step by step (each step reads the previous step's output from `data/processed/`):

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_select_researchers.py` | Pages OpenAlex authors by citations (CS-topic pool + robotics-topic pool). A researcher qualifies when at least half their topic counts are CS or robotics. Non-person and ghost profiles are dropped and split profiles merged. Output: `researchers.json` |
| 2 | `scripts/02_fetch_works.py [--limit N]` | For each researcher: 100 most-cited works (title, abstract, topic, authors) and joint-work counts with every co-author. Output: `papers`, `authorships`, `coauthors.parquet` |
| 3 | `scripts/03_embed.py` | SPECTER2 (`allenai/specter2_base` + proximity adapter) on title + abstract, on the GPU. The researcher vector is the log-citation-weighted mean of their papers, centred on the group mean |
| 4 | `scripts/04_build_graph.py` | 10-NN similarity edges, k-means communities (k by silhouette) grouped into 8 colour-coded areas, topic-based labels, UMAP layout. Exports GraphML/JSON |
| 5 | `scripts/05_load_neo4j.py` | Loads everything into Neo4j (`docker compose up -d` first) |
| 6 | `scripts/06_build_viz.py` | Writes `web/index.html` (and `web/atlas.html`, the same page without the html/head wrapper). Computes the 3D view's layout (3-D UMAP, cached in `data/processed/layout_3d.npz`) |

Every OpenAlex response is cached in `data/cache/`, so re-runs are free and an interrupted fetch
resumes where it stopped. Settings (number of researchers, works per researcher, k, robotics topic
list) live in `agraph/config.py`. `AGRAPH_N=200 ./run_all.sh` builds a smaller graph.

## Semantic search

```bash
.venv/bin/python scripts/search.py "legged locomotion with reinforcement learning"
.venv/bin/python scripts/search.py "graph neural networks" --neo4j    # via the Neo4j vector indexes
.venv/bin/python scripts/search.py --like "Sergey Levine"            # nearest research profiles
```

Text queries use SPECTER2's ad-hoc query adapter, which shares a vector space with the paper embeddings.

## Neo4j

`docker compose up -d` starts Neo4j 5.26 on localhost only: Browser at <http://localhost:7474>,
Bolt at `bolt://localhost:7687`. The user is `neo4j` and the password is `NEO4J_PASSWORD` in `.env`.

```
(:Researcher)-[:AFFILIATED_WITH {years, primary}]->(:Institution)
(:Researcher)-[:WORKS_ON {count}]->(:Topic)-[:IN_SUBFIELD]->(:Subfield)-[:IN_FIELD]->(:Field)-[:IN_DOMAIN]->(:Domain)
(:Researcher)-[:AUTHORED {position, top_work}]->(:Paper)-[:HAS_TOPIC]->(:Topic)
(:Researcher)-[:COAUTHOR {joint_works}]->(:Researcher)
(:Researcher)-[:SIMILAR_TO {score, rank}]->(:Researcher)
(:Researcher)-[:IN_COMMUNITY]->(:Community)-[:PART_OF]->(:Area)
vector indexes: researcher_embedding, paper_embedding (768-d, cosine) · fulltext index: paper_title
```

Neo4j's vector index reports cosine scores rescaled to `(1 + cos) / 2`, so 0.70 there is a raw
cosine of 0.40. `SIMILAR_TO.score` stores the raw cosine.

Example queries:

```cypher
// researchers with research most like Pieter Abbeel's (vector index)
MATCH (r:Researcher {name: 'Pieter Abbeel'})
CALL db.index.vector.queryNodes('researcher_embedding', 11, r.embedding) YIELD node, score
WHERE node <> r RETURN node.name, node.community_label, round(score, 3) AS score;

// collaborations that bridge research areas
MATCH (a:Researcher)-[c:COAUTHOR]->(b:Researcher) WHERE a.area <> b.area
RETURN a.name, a.area_label, b.name, b.area_label, c.joint_works ORDER BY c.joint_works DESC LIMIT 20;

// similar research but never co-authored: possible collaborators
MATCH (a:Researcher {name: 'Sergey Levine'})-[s:SIMILAR_TO]->(b)
WHERE NOT (a)-[:COAUTHOR]-(b) RETURN b.name, s.score ORDER BY s.score DESC;

// institutions with the most researchers in the atlas
MATCH (r:Researcher)-[:AFFILIATED_WITH {primary: true}]->(i) RETURN i.name, count(*) AS n ORDER BY n DESC LIMIT 15;

// papers on a subject (full text) and who in the atlas wrote them
CALL db.index.fulltext.queryNodes('paper_title', 'diffusion policy') YIELD node, score
MATCH (r:Researcher)-[:AUTHORED]->(node) RETURN node.title, node.year, collect(r.name) LIMIT 10;
```

## Method notes and caveats

- **"Most cited"** means total OpenAlex citations. CS citation counts are dominated by deep
  learning and vision, so researchers whose topics are mainly robotics are a minority (44 of the
  1,000 on the current data are tagged robotics: robotics share of topics ≥ 25%). Use the Robotics
  colour mode to see them. To force a robotics quota, change the selection in step 1.
- **Field test.** OpenAlex lists each author's top topics. A researcher counts as CS/robotics when
  CS topics plus the 16 robotics topics in `agraph/config.py` make up at least 50% of those topic counts.
- **Deep RL is not robotics.** OpenAlex files all deep RL (DQN, AlphaGo, ...) under the topic
  "Reinforcement Learning in Robotics". That topic counts toward someone's robotics share only if
  they also have a hands-on robotics topic (manipulation, SLAM, locomotion, ...). It is shown as
  "Reinforcement Learning" in map labels. Likewise "Topic Modeling" is where OpenAlex files most
  language-model papers, so it is ignored when naming groups.
- **Disambiguation.** OpenAlex sometimes splits one person into several profiles, or merges
  several people into one. Profiles are merged when they have the same name (or a spelling variant
  such as "A.Y. Ng" / "Andrew Y. Ng") and share an institution and at least one topic, or share
  most of their topics (Jiawei Han, Kenton Lee, Andrew Ng, Lotfi Zadeh, ...). A shared institution
  alone is not enough: two different "Hang Su"s at Tsinghua share none of their topics. Profiles
  with fewer than 5 works are folded into the matching full
  profile when there is exactly one ("D.P. Kingma" holding only the Adam paper → Diederik P.
  Kingma) and dropped otherwise. That removes ghost profiles built from one mega-paper (GPT-3,
  AlphaGo, PyTorch co-authors), and also Ming-Wei Chang, whose OpenAlex profile holds essentially
  only BERT. Other errors from the source remain.
- **Communities and areas.** Communities are k-means clusters of the researcher vectors
  (k = 14–24, chosen by cosine silhouette; 14 on the current data). Louvain on the similarity
  graph was tried first, but it lumped classical robotics and HPC into one community. After
  k-means, a community is split in two when its 2-way split is clearly separated (cosine
  silhouette ≥ 0.4, both parts ≥ 12 people), which keeps robotics apart from HCI. Communities
  are grouped into 8 areas (the most a categorical colour palette can keep distinguishable) by
  centroid similarity. A community that is at least 40% robotics always forms its own Robotics area.
  Areas are named by their most distinctive OpenAlex subfield, otherwise by their top topics.
- **Embedding.** Each researcher is described by their 100 most-cited works, so the vector
  reflects their best-known research rather than everything they have published.
- **Co-authorship** counts come from OpenAlex's aggregation over each researcher's full record
  (up to their 200 most frequent co-authors).
- **Measuring change over time.** Comparing "then" and "now" profiles is easy to get wrong: areas
  fitted on either period make that period look more distinct, and recent papers share a common
  direction that compresses differences. The findings fit reference areas on 2017–21 papers,
  centre each period on its own mean, and report shifts relative to the average researcher. An
  earlier version skipped these steps and wrongly concluded that CS was "converging on AI" and that
  robotics was "pulled hardest toward learning"; the Methods view lists the corrections.
- **Similarity vs collaboration.** Co-authors share papers, which makes their profiles similar by
  construction. The collaboration finding therefore rebuilds both profiles of every co-author pair
  without their shared papers before comparing similarity with co-authorship.

## Findings, Beyond CS and Methods

The atlas page has three more views next to Map and Table:

- **Findings** (`#finding-1` … `#finding-5`): map overlays and charts for five results in the CS
  graph: whether CS areas are converging (they are not, once measured fairly), the spread of LLM
  papers, how robotics moved relative to the rest of CS, collaboration vs research similarity, and
  one-paper researchers. Titles and claims are computed from the data.
- **Beyond CS** (`#beyond-cs`): the same question for 12 other fields (biology, neuroscience,
  medicine, chemistry, materials, physics, mathematics, earth sciences, economics, psychology,
  social sciences, humanities). It shows the AI share of all papers against the AI share of the
  most-cited papers per field (a field-level graph), per-field trends 2014–2025, top researchers
  then vs now with significance tests, and whether top-cited AI papers are about AI or use it.
- **Methods** (`#methods`): data, the embedding/graph pipeline, definitions of all 22 metrics,
  which metric to trust, caveats and how to reproduce.

```bash
.venv/bin/python analysis/fetch_recent.py     # each researcher's most-cited papers since 2022 (+ SPECTER2)
.venv/bin/python analysis/fields_fetch.py     # Beyond CS: researchers, top-cited papers, field counts (~$0.5 of OpenAlex budget)
.venv/bin/python analysis/fields_analyze.py   # Beyond CS numbers -> data/fields/results.json, analysis/fields_output.txt
.venv/bin/python analysis/export_findings.py  # findings 1-5, the single source of their numbers -> data/processed/findings.json
.venv/bin/python analysis/insights.py         # readable summary of findings.json -> analysis/insights_output.txt
.venv/bin/python scripts/06_build_viz.py      # rebuild the page (skips findings computed on a different graph)
```
