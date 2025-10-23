"""
Forge configuration file.
"""


HEAD="""

███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝                                      
""" # https://manytools.org/hacker-tools/ascii-banner/


from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional
from importlib import import_module

import os
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

import logging
log = logging.getLogger("forge")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(levelname)s | %(name)s | %(message)s')
    )
    log.addHandler(handler)


@dataclass
class VectorDBConfig:
    rebuild: bool = True
    stage_dir: Path = field(default_factory=lambda: Path(".forge"))
    data_dir: Path = field(default_factory=lambda: Path(".data"))
    index_type: str = field(default_factory=lambda: "hnsw")
    meta_index_path: Path = None # set in __post_init__
    embed_model: str = field(default=None) # | TODO: add support for cloud embedding model
    max_tokens: int = 510 # safe for all-MiniLM-L6-v2 limit (two-token room)
    overlap: float = field(init=False)
    overlap_ratio: float = field(default_factory=lambda: 0.1)
    batch_size: int = 32

    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50

    def __post_init__(self):
        self.overlap = int(self.max_tokens * self.overlap_ratio)
        self.stage_dir.mkdir(exist_ok=True)
        if self.meta_index_path is None:
            self.meta_index_path = self.stage_dir / "meta_vector.sqlite"
        if self.embed_model is None:
            self.embed_model = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class GraphConfig:
    stage_dir: Path = field(default_factory=lambda: Path(".forge"))
    graph_index_path: Path = None # set in __post_init__
    graph_meta_path: Path = None # set in __post_init__

    max_tokens: int = 2056 # | TODO

    tuple_delimiter: str = "|"
    record_delimiter: str = "##"
    completion_delimiter: str = "$$$"

    extraction_domains: list[str] = field(default_factory=lambda: [
        "base", # <- add more domains here
    ])
    extraction_templates: dict[str, dict] = field(default_factory=dict)
    entity_templates: dict = field(default_factory=dict)
    template_directory: str = ".forge._templates" # can be set to another dir


    entity_fields: list[str] = field(default_factory=lambda: [
        "<entity_name>",
        "<entity_type>",
        "<entity_description>",
    ])
    relationship_fields: list[str] = field(default_factory=lambda: [
        "<source_entity>",
        "<target_entity>",
        "<relationship_description>",
    ])
    
    system_prompt: str = "You are a helpful extraction assistant." # | TODO

    extraction_concurrency: Literal["sync", "async"] = "async"
    extraction_llm_backend: Literal["openai", "openrouter", "local"] = "openai"

    extraction_batch_size: int = 10

    def _load_extraction_templates(self):
        for domain in self.extraction_domains:
            try:
                module = import_module(f"{self.template_directory}.{domain}", package=__package__)
            except ImportError:
                raise ImportError(f"Expected template '{domain}' at {self.template_directory}/{domain}.py")
            try:
                self.extraction_templates[domain] = getattr(module, "EXTRACTION_TEMPLATE")
            except AttributeError:
                raise AttributeError(f"Expected EXTRACTION_TEMPLATE: str in {self.template_directory}/{domain}.py")
            try:
                self.entity_templates[domain] = getattr(module, "ENTITY_TYPES", [])
            except AttributeError:
                raise AttributeError(f"Expected ENTITY_TYPES: list[str] in {self.template_directory}/{domain}.py")

    def __post_init__(self):
        self.overlap = int(self.max_tokens * 0.10)
        self.stage_dir.mkdir(exist_ok=True)
        if self.graph_index_path is None:
            self.graph_index_path = self.stage_dir / "graph.sqlite"
        if self.graph_meta_path is None:
             self.graph_meta_path = self.stage_dir / "meta_graph.sqlite"
        self._load_extraction_templates()


@dataclass
class LLMConfig:
    sync_model: Optional[str] = "gpt-5-mini" # "qwen/qwen3-30b-a3b-2507"
    async_model: Optional[str] = "gpt-5-mini"
    semaphore_rate: Optional[int] = 1
    local_backend_url: Optional[str] = "http://localhost:1234/v1"

    price_million_input_tokens: Optional[float] = None # USD
    price_million_output_tokens: Optional[float] = None # USD
     
    api_key: str = None
    
    def __post_init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
