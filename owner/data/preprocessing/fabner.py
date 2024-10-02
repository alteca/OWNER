"""Preprocess FabNer to OWNER format
"""
from typing import List, Optional
import os
from datasets import load_dataset
from .base import BasePreprocessor
from ..serialization import serialize_owner_dataset
from ..model import Dataset, Document, Entity, Metadata


class FabNerPreprocessor(BasePreprocessor):
    """Preprocess FabNer dataset
    """

    def read_split(self, raw_data: list) -> Dataset:
        """Read FabNer split
        Args:
            raw_data (list): list of raw documents
        Returns:
            Dataset: parsed data
        """
        bio_tags = raw_data.features['ner_tags'].feature.names

        documents: List[Document] = []
        entity: Optional[Entity] = None

        for raw_document in raw_data:
            document = Document(id=raw_document['id'], sentences=[
                                raw_document['tokens']], entities=[])
            documents.append(document)

            for i, tag in enumerate(raw_document['ner_tags']):
                label = bio_tags[tag].lower()
                if label.startswith('e-'):
                    entity.end_word_idx = i
                    entity = None
                elif label.startswith('b-'):
                    entity = Entity(type=label[2:],
                                    sentence_idx=0,
                                    start_word_idx=i,
                                    end_word_idx=i+1)
                    document.entities.append(entity)
                elif label.startswith('s-'):
                    entity = Entity(type=label[2:],
                                    sentence_idx=0,
                                    start_word_idx=i,
                                    end_word_idx=i+1)
                    document.entities.append(entity)
                    entity = None

        entity_types = set()
        for document in documents:
            for entity in document.entities:
                entity_types.add(entity.type)
        metadata = Metadata(entity_types=entity_types)
        return Dataset(documents=documents, metadata=metadata)

    def preprocess_and_save(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

        dataset = load_dataset('DFKI-SLT/fabner')
        train_dataset = self.read_split(dataset['train'])
        serialize_owner_dataset(
            train_dataset, f'{output_folder}/train.json')

        dev_dataset = self.read_split(dataset['validation'])
        serialize_owner_dataset(
            dev_dataset, f'{output_folder}/dev.json')

        test_dataset = self.read_split(dataset['test'])
        serialize_owner_dataset(
            test_dataset, f'{output_folder}/test.json')
