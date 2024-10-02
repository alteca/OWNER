"""Common model classes
"""
from typing import List, Set
from pydantic import BaseModel, Field

# A sentence is a list of words
Sentence = List[str]


class Entity(BaseModel):
    """Entity
    """
    type: str | int         # Type of the entity
    sentence_idx: int       # Index of the sentence in the document
    # [start_word_idx, end_word_idx[
    start_word_idx: int
    end_word_idx: int


class MiniDocument(BaseModel):
    """Minimal document (for evaluation purpose)
    """
    id: str  # Id of the document (may differ from the index of the document in the dataset)
    entities: List[Entity] = Field(default=[])


class Document(MiniDocument):
    """Document
    """
    sentences: List[Sentence]


##### Dataset #####
class Metadata(BaseModel):
    """Metadata
    """
    entity_types: Set[str | int] = Field(default=set())


class Dataset(BaseModel):
    """Dataset
    """
    documents: List[Document]
    metadata: Metadata


class MiniDataset(BaseModel):
    """Dataset (for evaluation purpose)
    """
    documents: List[MiniDocument]
    metadata: Metadata
