"""Preprocess Mit formatted dataset (close to CoNLL) to convert into OWNER format
"""
import os
from typing import List, Optional
from .base import BasePreprocessor
from ..model import Document, Entity, Sentence, Metadata, Dataset
from ..serialization import serialize_owner_dataset


class MitPreprocessor(BasePreprocessor):
    """Preprocess Mit dataset
    """

    def __init__(self, config: dict):
        """Constructor
        Args:
            config (dict): data config
        """
        super().__init__(config)
        self.train_path = config['train_path']
        self.test_path = config['test_path']

    def read_split(self, path: str) -> Dataset:
        """Read Mit file
        Returns:
            Dataset: dataset.
        """
        documents: List[Document] = []
        entities: List[Entity] = []
        sentence: Sentence = []
        entity: Optional[Entity] = None
        i = 0

        with open(path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                line = line.strip()
                tsv = line.split('\t')

                if len(tsv) == 2:   # Text B-...
                    sentence.append(tsv[1])

                    label = tsv[0].lower()
                    if (label == 'o' or label.startswith('b-')) and entity is not None:
                        entity.end_word_idx = i
                        entity = None
                    if label.startswith('b-'):
                        entity = Entity(type=label[2:],
                                        sentence_idx=0,
                                        start_word_idx=i,
                                        end_word_idx=i)
                        entities.append(entity)
                    i += 1
                elif len(tsv) == 1:  # End of document
                    if entity is not None:
                        entity.end_word_idx = len(sentence)
                        entity = None
                    if len(sentence) > 0:
                        documents.append(
                            Document(id=str(len(documents)), sentences=[
                                     sentence], entities=entities))
                        sentence = []
                        entities = []
                    i = 0
                else:
                    raise SyntaxError(f'Unknown line format: {line}')

        if entity is not None:
            entity.end_word_idx = len(sentence)
        if len(sentence) > 0:
            documents.append(Document(id=str(len(documents)),
                             sentences=[sentence], entities=entities))

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

        test_dataset = self.read_split(self.test_path)
        serialize_owner_dataset(
            test_dataset, f'{output_folder}/test.json')
