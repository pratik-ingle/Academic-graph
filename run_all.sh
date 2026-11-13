#!/usr/bin/env bash
# Rebuild everything: select -> fetch -> embed -> graph -> Neo4j -> analysis -> web page.
# OpenAlex responses are cached in data/cache, so re-runs cost no API budget.
#   ANALYSIS=0 ./run_all.sh   skip the findings / Beyond CS analysis (page shows Map, Table, Methods)
# The first Beyond CS run (analysis/fields_fetch.py) uses about $0.5 of the daily OpenAlex budget.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -f .env ]; then
  echo "Missing .env: copy .env.example to .env and fill in OPENALEX_API_KEY and NEO4J_PASSWORD" >&2; exit 1
fi
set -a; source .env; set +a
: "${NEO4J_USER:=neo4j}"

$PY scripts/01_select_researchers.py
$PY scripts/02_fetch_works.py
$PY scripts/03_embed.py
$PY scripts/04_build_graph.py

docker compose up -d
echo "waiting for Neo4j to accept queries ..."
ready=0
for _ in $(seq 1 90); do
  if docker exec a-graph-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
if [ "$ready" != 1 ]; then echo "Neo4j did not become ready; check: docker logs a-graph-neo4j" >&2; exit 1; fi
$PY scripts/05_load_neo4j.py

if [ "${ANALYSIS:-1}" = "1" ]; then
  $PY analysis/fetch_recent.py
  $PY analysis/fields_fetch.py
  $PY analysis/fields_analyze.py > analysis/fields_output.txt
  $PY analysis/export_findings.py          # uses the Beyond CS counts for the CS-wide LLM trend
  $PY analysis/insights.py > analysis/insights_output.txt
fi

$PY scripts/06_build_viz.py
echo "Done. Open web/index.html, or Neo4j Browser at http://localhost:7474"
