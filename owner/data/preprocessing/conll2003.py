"""Preprocess CoNLL'03 formatted dataset to convert into OWNER format
"""
import os
from typing import List, Optional
from .base import BasePreprocessor
from ..model import Document, Entity,  Metadata, Dataset
from ..serialization import serialize_owner_dataset


class Conll2003Preprocessor(BasePreprocessor):
    """Preprocess CoNLL'03 dataset
    """

    def __init__(self, config: dict):
        """Constructor
        Args:
            config (dict): data config
        """
        super().__init__(config)
        self.train_path = config['train_path']
        self.dev_path = config['dev_path']
        self.test_path = config['test_path']

    def read_split(self, path: str) -> Dataset:
        """Read CoNLL'03 file
        Returns:
            Dataset: dataset.
        """
        documents: List[Document] = []
        entity: Optional[Entity] = None

        i = 0

        with open(path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                document = documents[-1] if len(documents) > 0 else None

                line = line.strip()
                tsv = line.split(' ')

                if line == '-DOCSTART- -X- -X- O':  # New document
                    document = Document(
                        id=str(len(documents)), sentences=[], entities=[])
                    documents.append(document)
                    entity = None
                elif len(tsv) == 4:   # Text ... ... B-...
                    # Add token to sentence
                    document.sentences[-1].append(tsv[0])

                    label = tsv[3].lower()
                    if (label == 'o' or label.startswith('b-')) and entity is not None:
                        entity.end_word_idx = i
                        entity = None
                    if label.startswith('b-'):
                        entity = Entity(type=label[2:],
                                        sentence_idx=len(
                                            document.sentences)-1,
                                        start_word_idx=i,
                                        end_word_idx=i)
                        document.entities.append(entity)
                    i += 1
                elif len(tsv) == 1:  # End of sentence
                    if entity is not None:
                        entity.end_word_idx = len(document.sentences[-1])
                        entity = None
                    i = 0
                    document.sentences.append([])
                else:
                    raise SyntaxError(f'Unknown line format: {line}')

        if entity is not None:
            entity.end_word_idx = len(document.sentences[-1])

        for document in documents:
            document.sentences = [
                sentence for sentence in document.sentences if len(sentence) > 0]

        # Metadata
        entity_types = set()
        for document in documents:
            for entity in document.entities:
                entity_types.add(entity.type)
        metadata = Metadata(entity_types=entity_types)
        return Dataset(documents=documents, metadata=metadata)

    def preprocess_and_save(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

        train_dataset = self.read_split(self.train_path)
        serialize_owner_dataset(
            train_dataset, f'{output_folder}/train.json')

        dev_dataset = self.read_split(self.dev_path)
        serialize_owner_dataset(
            dev_dataset, f'{output_folder}/dev.json')

        test_dataset = self.read_split(self.test_path)
        serialize_owner_dataset(
            test_dataset, f'{output_folder}/test.json')
