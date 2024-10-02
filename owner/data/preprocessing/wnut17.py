"""Preprocess Wnut17 to OWNER format
"""
import os
from datasets import load_dataset
from .base import BasePreprocessor
from ..serialization import serialize_owner_dataset
from ..model import Dataset, Document, Entity, Metadata


class Wnut17Preprocessor(BasePreprocessor):
    """Preprocess Wnut17 dataset
    """
    BIO_TAGS = ["O", "B-corporation", "I-corporation", "B-creative-work", "I-creative-work",
                "B-group", "I-group", "B-location", "I-location", "B-person", "I-person",
                "B-product", "I-product"]

    def read_split(self, raw_data: list) -> Dataset:
        """Read Wnut17 split
        Args:
            raw_data (list): list of raw documents
        Returns:
            Dataset: parsed data
        """
        documents = []
        for raw_document in raw_data:
            document = Document(id=raw_document['id'], entities=[], sentences=[
                                raw_document['tokens']])
            # Entities
            entity: Entity = None
            for i_word, bio_tag_idx in enumerate(raw_document['ner_tags']):
                bio_tag = Wnut17Preprocessor.BIO_TAGS[bio_tag_idx].lower()
                if (bio_tag == 'o' or bio_tag.startswith('b-')) \
                        and entity is not None:  # End of entity
                    entity.end_word_idx = i_word
                    entity = None
                if bio_tag.startswith('b-'):
                    entity = Entity(type=bio_tag[2:], sentence_idx=0,
                                    start_word_idx=i_word, end_word_idx=i_word+1)
                    document.entities.append(entity)
            if entity is not None:
                entity.end_word_idx = i_word+1
            documents.append(document)

        entity_types = set([tag[2:].lower()
                           for tag in Wnut17Preprocessor.BIO_TAGS if tag != 'O'])
        metadata = Metadata(entity_types=entity_types)
        return Dataset(documents=documents, metadata=metadata)

    def preprocess_and_save(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

        dataset = load_dataset('wnut_17')
        train_dataset = self.read_split(dataset['train'])
        serialize_owner_dataset(
            train_dataset, f'{output_folder}/train.json')

        dev_dataset = self.read_split(dataset['validation'])
        serialize_owner_dataset(
            dev_dataset, f'{output_folder}/dev.json')

        test_dataset = self.read_split(dataset['test'])
        serialize_owner_dataset(
            test_dataset, f'{output_folder}/test.json')
