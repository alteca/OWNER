"""Preprocess PileNer to OWNER format
"""
import os
import re
import json
import itertools
from datasets import load_dataset
from nltk import wordpunct_tokenize
from tqdm.auto import tqdm
from langdetect import detect
from .base import BasePreprocessor
from ..serialization import serialize_owner_dataset
from ..model import Dataset, Document, Entity, Metadata

TYPE_EXTRACTOR = re.compile(r'^What describes (.+) in the text\?$')


def normalize_entity(entity: str) -> str:
    """Normalize entity for checking
    Args:
        entity (str): entity
    Returns:
        str: normalized entity
    """
    return ''.join(entity.lower().split())


LEFT_BOUNDARY_MATCHER = '[ \\(\\)\\[\\]\\{\\}?,;:.!…·"\'*«»‘’“”`$@#\\-—\\+\\/\\\\|_%£><=^&]'
RIGHT_BOUNDARY_MATCHER = f'([s]?{LEFT_BOUNDARY_MATCHER}|$)'

CUSTOM_REPLACE = {
    "xe2x80x9c": "\"",
    "xe2x80x9d": "\"",
    "\t": " ",
    "\n": " ",
    "\r": " ",
    "\x0b": " ",
    "\x0c": " ",
    "\xa0": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u3000": " ",
    "\u2003": " ",
    "\u202f": " ",
    "\u2002": " ",
    "\u2005": " "
}

CUSTOM_WORD_REPLACE = \
    {k: f' {k} ' for k in '()[]\{\}「」?,;:.!…·"\'*«»‘’“”`$@#-—+/\\|_%£><=^&'}

MULTI_SPACE = re.compile('[ ][ ]+')


def custom_text_process(text: str) -> str:
    text = text[6:]
    for k, v in CUSTOM_REPLACE.items():
        text = text.replace(k, v)
    for k, v in CUSTOM_WORD_REPLACE.items():
        text = text.replace(k, v)
    text = MULTI_SPACE.sub(' ', text)
    return text.strip()


def custom_word_tokenize(text: str) -> list:
    for k, v in CUSTOM_WORD_REPLACE.items():
        text = text.replace(k, v)
    return wordpunct_tokenize(text)


def clean_entity(text: str) -> str:
    for k, v in CUSTOM_REPLACE.items():
        text = text.replace(k, v)
    for k, v in CUSTOM_WORD_REPLACE.items():
        text = text.replace(k, v)
    text = MULTI_SPACE.sub(' ', text)
    return text.strip()


class PileNerPreprocessor(BasePreprocessor):
    """Preprocess PileNer dataset
    """

    def read_split(self, raw_data: list) -> Dataset:
        """Read PileNer split
        Args:
            raw_data (list): list of raw documents
        Returns:
            Dataset: parsed data
        """
        documents = []
        for document in tqdm(raw_data):
            document_id = document['id']
            conversations = document['conversations']
            sentence_text_cased = custom_text_process(
                conversations[0]['value'])
            sentence_cased = custom_word_tokenize(sentence_text_cased)
            sentence_text = sentence_text_cased.lower()
            sentence = custom_word_tokenize(sentence_text)
            try:
                language = detect(sentence_text)
                if language != 'en':
                    continue
            except:
                pass

            # Compute char to word translation table
            char_to_word = []

            i_word, i_char = 0, 0
            for char in sentence_text:
                char_to_word.append(i_word)
                if i_word < len(sentence) and char == sentence[i_word][i_char]:
                    if i_char == len(sentence[i_word])-1:
                        i_word += 1
                        i_char = 0
                    else:
                        i_char += 1
                else:
                    assert char in ' ', int(
                        char)

            char_to_word = [0] + char_to_word + [len(sentence)]
            sentence_text = ' ' + sentence_text + ' '
            assert len(sentence_text) == len(char_to_word)

            entities = []
            for i in range(2, len(conversations), 2):
                entity_type = TYPE_EXTRACTOR.match(
                    conversations[i]['value']).group(1).lower()
                try:
                    predicted_entities = json.loads(
                        conversations[i+1]['value'])
                except:
                    predicted_entities = []
                predicted_entities = [{'\n': '\\n'}.get(
                    e, e) for e in predicted_entities]

                for predicted_entity in predicted_entities:
                    predicted_entity = clean_entity(predicted_entity)
                    for match in re.finditer(LEFT_BOUNDARY_MATCHER + re.escape(predicted_entity).lower() + RIGHT_BOUNDARY_MATCHER, sentence_text):
                        start_pos = match.start() + 1
                        end_pos = start_pos + len(predicted_entity)
                        start_word_idx = char_to_word[start_pos]
                        end_word_idx = char_to_word[end_pos-1] + 1

                        entities.append(Entity(type=entity_type, sentence_idx=0,
                                               start_word_idx=start_word_idx,
                                               end_word_idx=end_word_idx))
            documents.append(Document(id=document_id, sentences=[
                sentence_cased], entities=entities))

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
                    # Rules: keep the biggest one (in term of span length)
                    span1 = entity1.end_word_idx - entity1.start_word_idx
                    span2 = entity2.end_word_idx - entity2.start_word_idx
                    if span1 > span2:
                        entity2.sentence_idx = -1
                    else:
                        entity1.sentence_idx = -1
            document.entities = [
                entity for entity in document.entities if entity.sentence_idx != -1]
        return dataset

    def preprocess_and_save(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

        dataset = load_dataset('Universal-NER/Pile-NER-type')
        train_dataset = self.read_split(dataset['train'])
        train_dataset = self.filter_nested(train_dataset)
        serialize_owner_dataset(
            train_dataset, f'{output_folder}/train.json')
