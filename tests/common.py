from ..src.build import VectorDBBuilder, GraphBuilder
from ..src.query import VectorQueryEngine, GraphQueryEngine
from ..src.state import GraphIndex
from ..config import GraphConfig, VectorDBConfig, LLMConfig
from dataclasses import dataclass
from pathlib import Path

import logging
TEST_LOG = logging.getLogger("nexus-test")
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
LLM_CONFIG = LLMConfig()
