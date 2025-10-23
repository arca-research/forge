# FORGE

forge is a lightweight knowledge graph + vectorDB embedded library.

<p align="center">
  <img src="https://github.com/arca-research/forge/blob/main/forge/forge.png" width="450"/>
</p>
<p align="center">
        <b>VERSION 0.1</b>&nbsp
<br>

### quickstart

1. download requirements.txt
2. create .env and set OPENAI_API_KEY
3. tune config.py to your liking

Below is a sample script `main.py` (note that `main.py` and `forge` are at the same level)

```python
from forge import GraphBuilder, GraphQueryEngine, GraphIndex, GraphConfig

@dataclass
class DocData:
    """Container for document information."""
    document_id: int
    filepath: Path | str

fp = "data/myfile.txt"

# A) Building ---
# optional: see what the prompt looks like
with open(fp, "r", encoding="utf-8") as f:
    data = f.read()
print(builder._build_extraction_prompt(data))

# create the document container for the file
doc = DocData(document_id=0, filepath=fp)

# build() accepts list[DocLike] (defined in _schemas)
docs = [doc]

# build() will run LLM extraction and build a graph db in .forge
builder.build(docs)

# B) Using the query engine ---
query_engine = GraphQueryEngine()
entities = query_engine.list_all_entities()
print(entities)
ent = query_engine.Entity("<ENTITY>") # replace <ENTITY> with desired
print(ent.claims)

#  C) manually upserting aliases
cfg = GraphConfig()
index = GraphIndex(index_path=cfg.graph_index_path)
index.upsert_alias("FBI", "Federal Bureau of Investigation")
```
