# Knowledge Graph UI

An Obsidian-inspired, read-only graph explorer for an existing `graph.sqlite` knowledge graph. The backend is powered by FastAPI and the project consumes the existing `GraphIndex` API for canonicalisation, claims, and relationships. The frontend uses Cytoscape.js (via CDN) for layout, interaction, and export features.

## How to run

1. **Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -V   # confirm 3.10+ recommended
```

2. **Install Python deps**

```bash
pip install requirements-ui.txt
```

> You **do not** need Node/npm/yarn. The UI loads Cytoscape.js from a CDN.

3. **Point to your existing SQLite index**

* Ensure you have an existing `graph.sqlite` (or whichever path you use).
* You’ll pass it via CLI or env. Example:

```bash
export GRAPH_INDEX_PATH="/absolute/path/to/graph.sqlite"
uvicorn forge.ui.app:app --reload --port 8099
```

*or*

```bash
uvicorn forge.ui.app:app --reload --port 8099 --env-file .env
# where .env contains:
# GRAPH_INDEX_PATH=/absolute/path/to/graph.sqlite
```

4. **Open the UI**

* Visit: `http://localhost:8099/`
* You should see the graph render automatically. If it doesn’t:

  * Check the server logs for the resolved `GRAPH_INDEX_PATH`.
  * Ensure the file is readable and contains your tables.

5. **Use it**

* Pan/zoom.
* Type entity names in search (aliases should auto-resolve).
* Click a node: side panel with its claims and related entities.
* Click an edge: side panel with relationship claims.
* Export PNG / Download JSON via top-right buttons.

## API overview

- `GET /api/graph/snapshot` — full graph snapshot (nodes, edges, adjacency).
- `GET /api/entity/{name}` — canonical entity details, claims, related entities.
- `GET /api/edge?src=...&tgt=...` — relationship claims between two entities.

All endpoints are currently read-only and operate exclusively through `GraphIndex`.

## Extras

In app.js, you can adjust colours for various entity types:
```js
const ENTITY_COLORS = {
    ORGANIZATION: "#6BCB77",
    GEO: "#EE6C4D",
    PERSON: "#B084CC",
    DEFAULT: "#9AA0A6",
  };
```