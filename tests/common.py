from ..forge.build import VectorDBBuilder, GraphBuilder
from ..forge.query import VectorQueryEngine, GraphQueryEngine
from ..forge.state import GraphIndex
from ..config import GraphConfig, VectorDBConfig
from dataclasses import dataclass
from pathlib import Path

import logging
TEST_LOG = logging.getLogger("forge-test")
TEST_LOG.setLevel(logging.INFO)
if not TEST_LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(levelname)s | %(name)s | %(message)s')
    )
    TEST_LOG.addHandler(handler)


@dataclass
class DocData:
    """Container for document information."""
    document_id: int
    filepath: Path | str

GRAPH_CONFIG = GraphConfig()
GRAPH_BUILDER = GraphBuilder(debug=True)
