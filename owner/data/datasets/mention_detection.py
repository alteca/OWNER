"""Dataset for Mention Detection
"""
from typing import TypedDict
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from ..model import Dataset
from ...utils.pytorch import IGNORE_VALUE


class MdInstance(TypedDict):
    """Model for dataset instance
    """
    # Document, sentence and word indices
    document_idx: int
    sentence_idx: int
    offset: int
    word_ids: torch.Tensor      # Shape [max_len]

    input_ids: torch.Tensor     # Shape [max_len]
    attention_mask: torch.Tensor  # Shape [max_len]

    # Ignored tokens ([CLS], [SEP], [PAD], second and following tokens of word)
    ignored: torch.Tensor       # Shape [max_len]

    # Bio labels
    labels: torch.Tensor        # Shape [max_len]


class MentionDetectionDataset(torch.utils.data.Dataset):
    """Dataset for BIO sequence labeling training and evaluation
    """
    BIO_TO_ID = {'o': 0, 'b': 1, 'i': 2}

    def __init__(self, dataset: Dataset, tokenizer: str, max_len: int):
        super().__init__()
        self.entity_types = dataset.metadata.entity_types
        self.type_to_id = {t: i for i, t in enumerate(
            sorted(list(self.entity_types)))}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len
        self.dataset = dataset
        self.rows = self.preprocess(dataset)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> MdInstance:
        row = self.rows[index]
        words = row['sentence']
        entities = row['entities']

        tokenized_input = self.tokenizer(text=words, add_special_tokens=True,
                                         padding='max_length', max_length=self.max_len,
                                         truncation=True, is_split_into_words=True,
                                         return_attention_mask=True, return_tensors="pt")

        # Translate OWNER format to BIO
        word_labels = torch.ones([len(words)]) * self.BIO_TO_ID['o']
        for entity in entities:
            for i in range(entity['start_word_idx'], entity['end_word_idx']):
                if i == entity['start_word_idx']:
                    word_labels[i] = self.BIO_TO_ID['b']
                else:
                    word_labels[i] = self.BIO_TO_ID['i']

        # Translate BIO to token level BIO
        # (taken from: https://huggingface.co/docs/transformers/tasks/token_classification)
        word_ids = tokenized_input.word_ids(batch_index=0)

        # IGNORE_VALUE is ignored by Pytorch cross entropy
        token_labels = torch.ones(
            [self.max_len], dtype=torch.long) * IGNORE_VALUE
        # We ignore all tokens except the first token of each word/punctuation
        ignored_tokens = torch.ones([self.max_len], dtype=torch.bool)
        word_to_token = []
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                token_labels[i] = word_labels[word_idx]
                ignored_tokens[i] = 0
                word_to_token.append(i)
            previous_word_idx = word_idx

        word_ids = torch.tensor(
            [id if id is not None else IGNORE_VALUE for id in word_ids], dtype=torch.int)
        return {
            'input_ids': tokenized_input['input_ids'].flatten(),
            'attention_mask': tokenized_input['attention_mask'].flatten(),
            'ignored': ignored_tokens,
            'labels': token_labels,
            'word_ids': word_ids,
            'document_idx': row['document_idx'],
            'sentence_idx': row['sentence_idx'],
            'offset': row['offset']
        }

    def preprocess(self, dataset: Dataset) -> list:
        """Prepare OWNER dataset
        Args:
            dataset (Dataset): dataset
        Returns:
            list: list of instances
        """
        rows = []
        for document_idx, document in enumerate(tqdm(dataset.documents)):
            for sentence_idx, sentence in enumerate(document.sentences):
                tokenized_sentence = self.tokenizer(
                    sentence, add_special_tokens=False, is_split_into_words=True)
                word_ids = tokenized_sentence.word_ids(batch_index=0)
                reversed_ids = list(reversed(word_ids))

                start_window_idx = 0
                while start_window_idx < len(word_ids):
                    end_window_idx = start_window_idx + self.max_len - 2

                    start_window_word = word_ids[start_window_idx]

                    if end_window_idx >= len(word_ids):
                        end_window_idx = len(word_ids)
                        end_window_word = len(sentence)
                    elif word_ids[end_window_idx-1] == word_ids[end_window_idx]:
                        end_window_word = word_ids[end_window_idx] - 1
                        end_window_idx = len(word_ids) - \
                            reversed_ids.index(end_window_word)
                    else:
                        end_window_word = word_ids[end_window_idx]

                    curr_sentence = sentence[start_window_word:end_window_word]
                    curr_sentence_len = len(self.tokenizer(
                        curr_sentence, add_special_tokens=True, is_split_into_words=True)['input_ids'])
                    assert curr_sentence_len <= self.max_len, curr_sentence_len
                    entities = []
                    for entity in document.entities:
                        if entity.sentence_idx == sentence_idx and entity.start_word_idx >= start_window_word and entity.end_word_idx <= end_window_word:
                            entities.append({
                                'start_word_idx': entity.start_word_idx - start_window_word,
                                'end_word_idx': entity.end_word_idx - start_window_word,
                            })
                    rows.append({
                        'sentence': curr_sentence,
                        'entities': entities,
                        'document_idx': document_idx,
                        'sentence_idx': sentence_idx,
                        'offset': start_window_word
                    })

                    start_window_idx = end_window_idx
        return rows
