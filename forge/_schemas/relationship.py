from dataclasses import dataclass

@dataclass
class RelationshipRecord:
    source_name: str
    target_name: str
    strength: float = 0.0
    directed: bool = False