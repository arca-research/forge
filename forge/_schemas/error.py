"""
Error schemas across forge
"""

from typing import Optional

class RelationshipCollisionError(Exception):
    """Raised when a relationship source and target resolve to the same canonical entity."""
    
    def __init__(self, source: str, target: str, message: Optional[str] = None):
        self.source = source
        self.target = target
        if message is None:
            message = f"Collision detected: source '{source}' and target '{target}' resolve to the same canonical entity."
        super().__init__(message)


class EntityNotFoundError(Exception):
    """Raised when an entity cannot be found in the graph."""
    
    def __init__(self, entity_name: str, message: Optional[str] = None):
        self.entity_name = entity_name
        if message is None:
            message = f"Entity '{entity_name}' not found in graph."
        super().__init__(message)


class AliasConflictError(Exception):
    """Raised when an alias conflicts with existing entity or alias mappings."""
    
    def __init__(self, alias: str, existing_entity: str, new_entity: str, message: Optional[str] = None):
        self.alias = alias
        self.existing_entity = existing_entity
        self.new_entity = new_entity
        if message is None:
            message = (
                f"Alias conflict: '{alias}' is already mapped to '{existing_entity}', "
                f"cannot map to '{new_entity}'."
            )
        super().__init__(message)


class RelationshipMergeConflict(Exception):
    """Raised when merging entities would create duplicate relationships."""
    
    def __init__(self, canonical_entity: str, alias_entity: str):
        super().__init__(f"Merge conflict: self-loop between '{canonical_entity}' and '{alias_entity}', ")


class DeletionConflict(Exception):
    """Raised when deleting a table entry would result in a conflict"""
    
    def __init__(self, name: str, table: str, message: Optional[bool] = None):
        self.name = name
        self.table = table
        if message is None:
            message = f"Cannot delete '{name}' from '{table}'."
        super().__init__(message)


class RelationshipNotFoundError(Exception):
    def __init__(self, src: str, tgt: str, directed: Optional[bool]):
        dir_str = "directed" if directed is True else "undirected" if directed is False else "any"
        super().__init__(f"No {dir_str} relationship found between '{src}' and '{tgt}'")