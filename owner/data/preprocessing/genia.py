"""Preprocess Genia to OWNER format
"""
import os
from datasets import load_dataset
from .base import BasePreprocessor
from ..serialization import serialize_owner_dataset
from ..model import Dataset, Document, Entity, Metadata


class GeniaPreprocessor(BasePreprocessor):
    """Preprocess Genia dataset
    """

    def read_split(self, raw_data: list) -> Dataset:
        """Read Genia split
        Args:
            raw_data (list): list of raw documents
        Returns:
            Dataset: parsed data
        """
        documents = []
        current_id = -1
        current_sentence = 0
        for raw_document in raw_data:
            if raw_document['org_id'] != current_id:
                current_id = raw_document['org_id']
                documents.append(
                    Document(id=current_id, sentences=[], entities=[]))
                current_sentence = 0

            document = documents[-1]
            document.sentences.append(raw_document['tokens'])
            for entity in raw_document['entities']:
                document.entities.append(Entity(
                    type=entity['type'], sentence_idx=current_sentence,
                    start_word_idx=entity['start'], end_word_idx=entity['end']))
            current_sentence += 1

        entity_types = set()
        for document in documents:
            for entity in document.entities:
                entity_types.add(entity.type)
        metadata = Metadata(entity_types=entity_types)
        return Dataset(documents=documents, metadata=metadata)

    def preprocess_and_save(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

        dataset = load_dataset('Rosenberg/genia')
        train_dataset = self.read_split(dataset['train'])
        serialize_owner_dataset(
            train_dataset, f'{output_folder}/train.json')

        dev_dataset = self.read_split(dataset['validation'])
        serialize_owner_dataset(
            dev_dataset, f'{output_folder}/dev.json')

        test_dataset = self.read_split(dataset['test'])
        serialize_owner_dataset(
            test_dataset, f'{output_folder}/test.json')
