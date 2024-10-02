"""Preprocess Gum/Gentle formatted dataset to convert into OWNER format
"""
import re
import os
from typing import List
import itertools
from .base import BasePreprocessor
from ..model import Document, Entity, Metadata, Dataset
from ..serialization import serialize_owner_dataset


class GumPreprocessor(BasePreprocessor):
    """Preprocess Gum dataset
    """
    ignored_regex = re.compile(r'\([0-9]+\)')
    start_entity_regex = re.compile(r'\(([^()-]+)-([0-9]+)([^()]+)?')
    end_entity_regex = re.compile(r'([^()-]+)-([0-9]+)([^()]+)?\)')

    def __init__(self, config: dict):
        """Constructor
        Args:
            config (dict): data config
        """
        super().__init__(config)
        self.test_path = config['test_path']
        self.nested = config['nested']
        self.remove_abstract = config['remove_abstract']

    def read_document(self, path: str, document_id: str) -> Document:
        """Read Gum document file
        Args:
            path (str): path to document
            document_id (str): id of document
        Returns:
            Document: processed document
        """
        # Parse CoNLL document
        sentences = [[]]
        entities = {}
        i = 0
        with open(path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                if line.startswith('#'):
                    continue

                tsv = line.split('\t')
                if len(tsv) == 1:
                    continue

                token = tsv[1].strip()
                sentences[-1].append(token)

                entity_types = tsv[2].strip()
                if entity_types == '_' or GumPreprocessor.ignored_regex.match(entity_types):
                    pass
                else:
                    # Entity opening
                    for match in GumPreprocessor.start_entity_regex.finditer(entity_types):
                        entity_type = match.group(1).lower()
                        document_id = match.group(2)
                        entity = {
                            'type': entity_type,
                            'start_sentence_idx': len(sentences)-1,
                            'start_word_idx': i,
                            'end_word_idx': i+1
                        }
                        entities[document_id] = entity

                    # Entity closing
                    for match in GumPreprocessor.end_entity_regex.finditer(entity_types):
                        entity_type = match.group(1).lower()
                        document_id = match.group(2)
                        entity = entities[document_id]
                        entity['end_sentence_idx'] = len(sentences)-1
                        entity['end_word_idx'] = i + 1
                i += 1

                if token in ['.', '?', '!', '...', '…']:
                    sentences.append([])
                    i = 0
        entities = list(entities.values())

        # Check if each entity is in a single sentence, else merge sentences
        finished = False
        while not finished:
            finished = True

            for entity in entities:
                if entity['start_sentence_idx'] < entity['end_sentence_idx']:
                    finished = False
                    sentence_idx = entity['start_sentence_idx']
                    break

            if finished:
                break

            # Merge sentence
            word_idx_offset = len(sentences[sentence_idx])
            sentences[sentence_idx].extend(sentences[sentence_idx+1])
            sentences.pop(sentence_idx+1)

            # Modify each entity
            for entity in entities:
                if entity['start_sentence_idx'] == sentence_idx + 1:
                    entity['start_sentence_idx'] = sentence_idx
                    entity['start_word_idx'] += word_idx_offset
                elif entity['start_sentence_idx'] > sentence_idx + 1:
                    entity['start_sentence_idx'] -= 1

                if entity['end_sentence_idx'] == sentence_idx + 1:
                    entity['end_sentence_idx'] = sentence_idx
                    entity['end_word_idx'] += word_idx_offset
                elif entity['end_sentence_idx'] > sentence_idx + 1:
                    entity['end_sentence_idx'] -= 1

        # Remove empty sentences
        i_sentence = 0
        while i_sentence < len(sentences):
            if len(sentences[i_sentence]) == 0:
                sentences.pop(i_sentence)

                # Correct sentence idx for entities
                for entity in entities:
                    if entity['start_sentence_idx'] >= i_sentence:
                        entity['start_sentence_idx'] -= 1
            else:
                i_sentence += 1

        for entity in entities:
            assert entity['start_sentence_idx'] == entity['end_sentence_idx'] and \
                entity['start_word_idx'] < entity['end_word_idx']

        if self.remove_abstract:
            entities = [
                entity for entity in entities if entity['type'] != 'abstract']

        return Document(
            id=document_id,
            sentences=sentences,
            entities=[Entity(type=entity['type'],
                             sentence_idx=entity['start_sentence_idx'],
                             start_word_idx=entity['start_word_idx'],
                             end_word_idx=entity['end_word_idx'])
                      for entity in entities]
        )

    def read_split(self, path: str) -> Dataset:
        """Read Gum folder
        Returns:
            Dataset: dataset.
        """
        documents: List[Document] = []
        for filename in os.listdir(path):
            documents.append(self.read_document(
                f'{path}/{filename}', filename))

        # Metadata
        entity_types = set()
        for document in documents:
            for entity in document.entities:
                entity_types.add(entity.type)
        metadata = Metadata(entity_types=entity_types)
        return Dataset(documents=documents, metadata=metadata)

    def filter_nested(self, dataset: Dataset) -> Dataset:
        """Filter nested entities
        Warning: modify in place
        Args:
            dataset (Dataset): dataset
        Returns:
            Dataset: filtered dataset.
        """

        for document in dataset.documents:
            for entity1, entity2 in itertools.combinations(document.entities, 2):
                if entity1.sentence_idx == entity2.sentence_idx and \
                        entity1.start_word_idx < entity2.end_word_idx and \
                        entity2.start_word_idx < entity1.end_word_idx:  # Entities are nested
                    # Rules
                    # - if one entity is abstract and the other is not abstract: keep non abstract entity
                    # - else keep the smallest one (in term of span length)
                    if entity1.type == 'abstract' and entity2.type != 'abstract':
                        entity1.sentence_idx = -1
                    elif entity1.type != 'abstract' and entity2.type == 'abstract':
                        entity2.sentence_idx = -1
                    else:
                        span1 = entity1.end_word_idx - entity1.start_word_idx
                        span2 = entity2.end_word_idx - entity2.start_word_idx
                        if span1 < span2:
                            entity2.sentence_idx = -1
                        else:
                            entity1.sentence_idx = -1
            document.entities = [
                entity for entity in document.entities if entity.sentence_idx != -1]
        return dataset

    def preprocess_and_save(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

        test_dataset = self.read_split(self.test_path)
        if not self.nested:
            test_dataset = self.filter_nested(test_dataset)
        serialize_owner_dataset(
            test_dataset, f'{output_folder}/test.json')
