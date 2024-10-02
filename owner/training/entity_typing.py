"""Trainer for Entity Typing
"""
from typing import cast, Dict, List
import logging
import mlflow
from tqdm.auto import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from transformers import get_linear_schedule_with_warmup
from matplotlib import pyplot as plt
from ..utils.pytorch import get_num_workers
from ..evaluation.base import convert_document_to_entities
from ..evaluation.entity_typing import evaluate_entity_typing
from ..models.entity_typing import EntityEncodingModel, AutoKmeans
from ..data.serialization import from_owner
from ..data.model import Dataset, MiniDocument, Entity
from ..data.datasets.entity_typing import EntityTypingDataset
from .base import BaseTrainer
logger = logging.getLogger('mlflow')


class BatchTripletMarginLoss(nn.Module):
    """Triplet margin loss with batch triplet extraction
    """

    def __init__(self, margin: float = 1.0):
        """Constructor
        Args:
            margin (float, optional): margin.
        """
        super().__init__()
        self._margin = margin

    def forward(self, entity_types: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute triplet margin loss between all valid triplets in batch
        Args:
            entity_types (torch.Tensor): entity types. Shape [batch]
            embeddings (torch.Tensor): entity embeddings. Shape [batch, 768]
        Returns:
            torch.Tensor: loss
        """
        type_coherence = (entity_types.unsqueeze(dim=1) -
                          entity_types.unsqueeze(dim=0)) == 0
        valid_triplets = (type_coherence.unsqueeze(dim=2) * ~
                          type_coherence.unsqueeze(dim=1)).detach()

        distances = torch.cdist(embeddings, embeddings, p=2)
        triplet_distances = (distances.unsqueeze(
            dim=2) - distances.unsqueeze(dim=1)) * valid_triplets

        loss = torch.clamp(triplet_distances + self._margin, 0.).sum()

        return loss / valid_triplets.sum()


class EntityTypingTrainer(BaseTrainer):
    """Entity Typing trainer
    """

    def __init__(self, config: dict):
        """Constructor
        Args:
            config (dict): model args
        """
        super().__init__(config)
        et_config = config['entity_typing']

        self.model = self.accelerator.prepare(
            EntityEncodingModel(et_config['plm_name']))

        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler.LambdaLR = None
        self.loss_fn: BatchTripletMarginLoss = None
        self.train_dataset: EntityTypingDataset = None
        self.test_dataset: EntityTypingDataset = None

    def load_data(self, training: bool = True):
        data_config = self.config['data']
        et_config = self.config['entity_typing']
        tokenizer = et_config['plm_name']
        max_len = et_config['max_len']
        template = et_config['template']

        if training:
            train_dataset = from_owner(
                data_config['train_dataset_path'],
                data_config['train_dataset_name']
            )
            mlflow.log_input(train_dataset, context='et_train')
            self.train_dataset = EntityTypingDataset(
                train_dataset.dataset, tokenizer, max_len, template)

        test_dataset = from_owner(
            data_config['test_dataset_path'],
            data_config['test_dataset_name']
        )
        mlflow.log_input(test_dataset, context='et_test')
        self.test_dataset = EntityTypingDataset(
            test_dataset.dataset, tokenizer, max_len, template)

    def train(self):
        logger.info("Training Entity Typing")
        et_config = self.config['entity_typing']
        batch_size = et_config['batch_size']
        num_epochs = et_config['num_epochs']
        num_workers = get_num_workers()

        train_dataloader = self.accelerator.prepare(DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers))
        test_dataloader = self.accelerator.prepare(DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers))
        num_training_steps = num_epochs * len(train_dataloader)

        self.optimizer = self.accelerator.prepare(optim.AdamW(
            self.model.parameters(), et_config['learning_rate']))
        self.scheduler = self.accelerator.prepare(get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps))
        self.loss_fn = self.accelerator.prepare(BatchTripletMarginLoss())

        epoch = 0
        self.evaluate_dataloader(test_dataloader, 'test_epoch', 0)
        for epoch in range(1, num_epochs+1):
            logger.info('Epoch %s/%s', epoch, num_epochs)
            self.train_one_epoch(epoch, train_dataloader, test_dataloader)
            epoch += 1

    def train_one_epoch(self, epoch: int, train_dataloader: DataLoader, test_dataloader: DataLoader):
        """Train model for one epoch

        Args:
            epoch (int): epoch
        """
        train_dataloader_len = len(train_dataloader)

        self.model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=train_dataloader_len):
            current_step = (epoch-1) * train_dataloader_len + i
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            mask_index = batch['mask_index']
            entity_type_labels = batch['entity_type_label']

            mask_embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask, mask_index=mask_index)

            self.optimizer.zero_grad()
            loss = self.loss_fn(entity_type_labels, mask_embeddings)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

            mlflow.log_metrics({
                'et_train_batch_triplet-margin-loss': loss
            }, step=current_step)
        self.model.eval()

        self.evaluate_dataloader(
            test_dataloader, step=current_step, context="test_epoch")

    def evaluate_dataloader(self, dataloader: DataLoader, context: str,
                            step: int = None, fast: bool = True, prefix: str = 'et'):
        """Evaluate dataloader
        Args:
            dataloader (DataLoader): dataloader
            context (str): context
            step (int, optional): current step. Defaults to None.
            fast (bool, optional): run fast evaluation (without estimating k).
                Defaults to True.
        """
        dataset: Dataset = cast(EntityTypingDataset,
                                dataloader.dataset).dataset

        y_true: Dict[int, MiniDocument] = {}
        y_pred: Dict[int, MiniDocument] = {}

        entity_embeddings = []
        entity_type_labels = []
        entity_metadata = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                mask_index = batch['mask_index']
                curr_entity_type_labels = batch['entity_type_label']
                documents_ids = batch['document_idx']
                sentence_ids = batch['sentence_idx']
                start_word_ids = batch['start_word_idx']
                end_word_ids = batch['end_word_idx']

                curr_entity_embeddings = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, mask_index=mask_index)
                entity_embeddings.append(curr_entity_embeddings.detach().cpu())
                entity_type_labels.append(
                    curr_entity_type_labels.detach().cpu())

                for i, document_idx in enumerate(documents_ids):
                    document_idx = document_idx.item()
                    if document_idx not in y_true:
                        y_true[document_idx] = convert_document_to_entities(
                            dataset.documents[document_idx])
                    entity_metadata.append({
                        'document_idx': document_idx,
                        'sentence_idx': sentence_ids[i].item(),
                        'start_word_idx': start_word_ids[i].item(),
                        'end_word_idx': end_word_ids[i].item(),
                    })

            # Predict clustering
            entity_embeddings = torch.cat(entity_embeddings, dim=0)
            entity_type_labels = torch.cat(entity_type_labels, dim=0)

            if fast:
                clustering_model = KMeans(
                    n_clusters=len(dataset.metadata.entity_types),
                    random_state=self.config['seed'], n_init=10)
                clustering_model.fit(entity_embeddings)
            else:
                et_config = self.config['entity_typing']
                clustering_model = AutoKmeans(et_config['k_min'],
                                              et_config['k_max'],
                                              et_config['k_step'],
                                              seed=self.config['seed'])
                clustering_model.fit(entity_embeddings)

            y_pred_clusters = clustering_model.predict(
                entity_embeddings).tolist()

            for metadata, cluster in zip(entity_metadata, y_pred_clusters):
                document_idx = metadata['document_idx']
                if document_idx not in y_pred:
                    y_pred[document_idx] = MiniDocument(
                        id=dataset.documents[document_idx].id, entities=[])

                entity = Entity(
                    sentence_idx=metadata['sentence_idx'],
                    start_word_idx=metadata['start_word_idx'],
                    end_word_idx=metadata['end_word_idx'], type=cluster)
                y_pred[document_idx].entities.append(entity)

            # Add documents without entities
            for document_idx, document in enumerate(dataset.documents):
                if document_idx not in y_pred:
                    y_pred[document_idx] = MiniDocument(
                        id=document.id, entities=[])
                if document_idx not in y_true:
                    y_true[document_idx] = MiniDocument(
                        id=document.id, entities=document.entities)

            y_true = list(y_true.values())
            y_pred = list(y_pred.values())
            assert len(y_true) == len(
                y_pred), f'Not the same number of documents {len(y_true)} != {len(y_pred)}'

            evaluate_entity_typing(
                y_true, y_pred, dataset.metadata.entity_types, None, context, step, prefix)

    def evaluate(self):
        logger.info("Evaluating Entity Typing")
        et_config = self.config['entity_typing']
        batch_size = et_config['batch_size']
        num_workers = get_num_workers()
        test_dataloader = self.accelerator.prepare(DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers))

        self.evaluate_dataloader(test_dataloader, "test", fast=False)

    def predict_dataset(self, dataset: Dataset, context: str,
                        step: int = None, prefix: str = 'et') -> List[MiniDocument]:
        """Predict dataset
        Args:
            dataset (Dataset): dataset
        Returns:
            List[MiniDocument]: predicted types
        """
        et_config = self.config['entity_typing']
        tokenizer = et_config['plm_name']
        max_len = et_config['max_len']
        template = et_config['template']
        batch_size = et_config['batch_size']
        num_workers = get_num_workers()

        dataloader = self.accelerator.prepare(DataLoader(
            EntityTypingDataset(dataset, tokenizer, max_len, template),
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        ))

        y_pred: Dict[int, MiniDocument] = {}

        entity_embeddings = []
        entity_metadata = []
        entity_type_labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                mask_index = batch['mask_index']
                documents_ids = batch['document_idx']
                curr_entity_type_labels = batch['entity_type_label']
                sentence_ids = batch['sentence_idx']
                start_word_ids = batch['start_word_idx']
                end_word_ids = batch['end_word_idx']

                curr_entity_embeddings = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, mask_index=mask_index)
                entity_embeddings.append(curr_entity_embeddings.detach().cpu())
                entity_type_labels.append(
                    curr_entity_type_labels.detach().cpu())

                for i, document_idx in enumerate(documents_ids):
                    document_idx = document_idx.item()
                    entity_metadata.append({
                        'document_idx': document_idx,
                        'sentence_idx': sentence_ids[i].item(),
                        'start_word_idx': start_word_ids[i].item(),
                        'end_word_idx': end_word_ids[i].item(),
                    })

            # Predict clustering
            entity_embeddings = torch.cat(entity_embeddings, dim=0)
            entity_type_labels = torch.cat(entity_type_labels, dim=0)
            et_config = self.config['entity_typing']
            clustering_model = AutoKmeans(et_config['k_min'],
                                          et_config['k_max'],
                                          et_config['k_step'],
                                          seed=self.config['seed'])
            clustering_model.fit(entity_embeddings)

            y_pred_clusters = clustering_model.predict(
                entity_embeddings).tolist()

            for metadata, cluster in zip(entity_metadata, y_pred_clusters):
                document_idx = metadata['document_idx']
                if document_idx not in y_pred:
                    y_pred[document_idx] = MiniDocument(
                        id=dataset.documents[document_idx].id, entities=[])

                entity = Entity(
                    sentence_idx=metadata['sentence_idx'],
                    start_word_idx=metadata['start_word_idx'],
                    end_word_idx=metadata['end_word_idx'], type=cluster)
                y_pred[document_idx].entities.append(entity)

            # Add documents without entities
            for document_idx, document in enumerate(dataset.documents):
                if document_idx not in y_pred:
                    y_pred[document_idx] = MiniDocument(
                        id=document.id, entities=[])

            y_pred = [y_pred[k] for k in sorted(y_pred)]
        return y_pred

    def save_model(self, folder: str):
        super().save_model(folder)
        torch.save(self.model.state_dict(), f'{folder}/entity_typing.pt')

    def load_model(self, folder: str):
        self.model.load_state_dict(torch.load(f'{folder}/entity_typing.pt'))
        self.model.eval()
