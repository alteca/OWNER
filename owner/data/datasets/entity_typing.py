"""Dataset for Entity Typing
"""
from typing import TypedDict, List
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from ..model import Dataset, Entity


class EtInstance(TypedDict):
    """Model for dataset instance
    """

    # Current entity identifier
    document_idx: int
    sentence_idx: int
    start_pos_idx: int
    end_pos_idx: int

    input_ids: torch.Tensor  # Shape [max_len]
    attention_mask: torch.Tensor  # Shape [max_len]

    # Index of masked word (for entity typing embedding)
    mask_index: int

    entity_type_label: int  # Entity type id


class EntityTypingDataset(torch.utils.data.Dataset):
    """Represents a dataset to iterate over data for entity embedding predictions
        and contrastive learning.
    """

    def __init__(self, dataset: Dataset, tokenizer: str, max_len: int, template: str):
        """Constructor

        Args:
            dataset (Dataset): dataset
            tokenizer (str): tokenizer name
            max_len (int): max length
            template (str): template to use
        """
        super().__init__()
        self.max_len = max_len
        self.template = template
        self.entity_type_to_id = {t: i for i, t in enumerate(
            sorted(list(dataset.metadata.entity_types)))}
        self.id_to_entity_type = {i: l for l,
                                  i in self.entity_type_to_id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mask_token_id = self.tokenizer.mask_token_id

        self.dataset = dataset
        self.rows = self.preprocess(dataset)

    def __len__(self) -> int:
        """Returns length of dataset
        Returns:
            int: length of dataset
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> EtInstance:
        """Returns item of dataset at index
        Args:
            index (int): index
        Returns:
            dict: information needed to compute entity embedding
        """
        row = self.rows[index]

        # Generate and encode prompt
        prompt = row['prompt']
        encoded_dict = self.tokenizer.encode_plus(
            prompt,
            max_length=self.max_len,
            padding='max_length',
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_dict['input_ids'].view(-1)
        attention_mask = encoded_dict['attention_mask'].view(-1)

        # Index of [MASK] token
        mask_index = (
            input_ids == self.mask_token_id).nonzero().flatten().item()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mask_index': mask_index,
            'document_idx': row['document_idx'],
            'sentence_idx': row['sentence_idx'],
            'start_word_idx': row['start_word_idx'],
            'end_word_idx': row['end_word_idx'],
            'entity_type_label': self.entity_type_to_id[row['entity_type']]
        }

    def get_prompt(self, sentence: List[str], entity: Entity) -> str:
        """Generate prompt for a given sentence and entity
        Args:
            sentence (List[str]): sentence
            entity (Entity): entity
        Returns:
            str: prompt
        """
        entity_text = ' '.join(
            sentence[entity.start_word_idx:entity.end_word_idx])
        prompt = self.template.format(
            sentence=' '.join(sentence), entity=entity_text)
        if len(self.tokenizer(prompt)['input_ids']) <= self.max_len:
            return prompt

        # Reduce sentence to embed entity
        empty_prompt = self.template.format(sentence='', entity=entity_text)
        empty_prompt_len = len(self.tokenizer(empty_prompt)['input_ids'])
        if empty_prompt_len > self.max_len // 2:  # Entity is too long
            raise Exception()

        tokenized_sentence = self.tokenizer(
            text=sentence, is_split_into_words=True, add_special_tokens=False)
        word_ids = tokenized_sentence.word_ids(batch_index=0)
        sentence_len = len(word_ids)
        target_sentence_len = self.max_len - empty_prompt_len - 1

        start_entity_idx = word_ids.index(entity.start_word_idx)
        end_entity_idx = len(
            word_ids) - list(reversed(word_ids)).index(entity.end_word_idx-1)+1
        half_window_size = (target_sentence_len -
                            (end_entity_idx - start_entity_idx)) // 2

        # Find correct token start and end to maximize embedding
        if start_entity_idx - half_window_size < 0:
            start_window_idx = 0
            end_window_idx = target_sentence_len
        elif end_entity_idx + half_window_size > sentence_len:
            start_window_idx = sentence_len - target_sentence_len
            end_window_idx = sentence_len
        else:
            start_window_idx = start_entity_idx - half_window_size
            end_window_idx = end_entity_idx + half_window_size

        # Adjust token to word boundaries (so that < max_len)
        if start_window_idx <= 0:
            start_window_word = 0
        elif word_ids[start_window_idx] == word_ids[start_window_idx-1]:
            start_window_word = word_ids[start_window_idx] + 1
        else:
            start_window_word = word_ids[start_window_idx]

        if end_window_idx >= sentence_len:
            end_window_word = len(sentence)
        elif word_ids[end_window_idx-1] == word_ids[end_window_idx]:
            end_window_word = word_ids[end_window_idx] - 1
        else:
            end_window_word = word_ids[end_window_idx]

        prompt = self.template.format(sentence=' '.join(
            sentence[start_window_word:end_window_word]), entity=entity_text)
        assert len(self.tokenizer(prompt)['input_ids']) <= self.max_len
        return prompt

    def preprocess(self, dataset: Dataset) -> list:
        """Prepare OWNER dataset for iteration
        Args:
            dataset (Dataset): dataset
        Returns:
            list: list of instances
        """
        # Generate prompts
        rows = []
        for document_idx, document in enumerate(tqdm(dataset.documents)):
            sentences = document.sentences
            for entity in document.entities:
                sentence_idx = entity.sentence_idx

                try:
                    prompt = self.get_prompt(
                        sentences[sentence_idx], entity)
                    rows.append({
                        'entity_type': entity.type,
                        'document_idx': document_idx,
                        'sentence_idx': sentence_idx,
                        'start_word_idx': entity.start_word_idx,
                        'end_word_idx': entity.end_word_idx,
                        'prompt': prompt
                    })
                except:  # No viable prompt
                    pass
        return rows
