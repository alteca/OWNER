"""Preprocess I2b2 formatted dataset to convert into OWNER format
"""
import os
from typing import List
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from .base import BasePreprocessor
from ..model import Document, Entity, Metadata, Dataset
from ..serialization import serialize_owner_dataset


class I2b2Preprocessor(BasePreprocessor):
    """Preprocess I2b2 dataset
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

    def read_document(self, path: str, document_id: str) -> Document:
        """Read i2b2 document file
        Args:
            path (str): path to document
            document_id (str): id of document
        Returns:
            Document: processed document
        """
        with open(path, 'r', encoding='utf-8') as file:
            xml = BeautifulSoup(file, 'xml')
        i2b2_root = xml.find('deIdi2b2')

        # Raw text and entities
        text = i2b2_root.find('TEXT').get_text()
        entities = []
        for candidate in i2b2_root.find('TAGS').children:
            tag = candidate.name
            if tag is not None:
                entities.append({
                    'type': candidate.get('TYPE').lower(),
                    'start':  int(candidate.get('start')),
                    'end':  int(candidate.get('end')),
                })

        # Split text in sentences and words
        sentences = [wordpunct_tokenize(sentence)
                     for sentence in sent_tokenize(text)]

        # Align word/sentences with document chars
        sentence_idx = 0
        word_idx = 0
        char_idx = 0
        pos_to_sent_word = []
        for char in text:
            pos_to_sent_word.append([sentence_idx, word_idx])
            if sentence_idx < len(sentences) and char == sentences[sentence_idx][word_idx][char_idx]:
                if char_idx < len(sentences[sentence_idx][word_idx]) - 1:
                    char_idx += 1
                elif word_idx < len(sentences[sentence_idx]) - 1:
                    word_idx += 1
                    char_idx = 0
                else:
                    sentence_idx += 1
                    word_idx = 0
                    char_idx = 0

        # Find entity positions
        for entity in entities:
            start_pos = pos_to_sent_word[entity['start']]
            end_pos = pos_to_sent_word[entity['end']-1]
            entity['start_sentence_idx'] = start_pos[0]
            entity['start_word_idx'] = start_pos[1]
            entity['end_sentence_idx'] = end_pos[0]
            entity['end_word_idx'] = end_pos[1]+1

        for entity in entities:
            assert entity['start_sentence_idx'] < len(sentences)
            if entity['end_sentence_idx'] >= len(sentences):
                entity['end_sentence_idx'] = len(sentences)-1
                entity['end_word_idx'] = len(sentences[-1])

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

        for entity in entities:
            assert entity['start_sentence_idx'] == entity['end_sentence_idx'] and \
                entity['start_word_idx'] < entity['end_word_idx']

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
        """Read I2b2 folder
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
