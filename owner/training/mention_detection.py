"""Trainer for BIO sequence labeling mention detection
"""
from typing import cast, Dict, List
import logging
import mlflow
from tqdm.auto import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from ..data.model import Dataset, MiniDocument
from ..data.serialization import from_owner
from ..data.datasets.mention_detection import MentionDetectionDataset
from ..models.mention_detection import MdBioModel
from ..utils.pytorch import get_num_workers, IGNORE_VALUE
from ..evaluation.mention_detection import evaluate_mention_detection, \
    convert_bio_to_entities
from ..evaluation.base import convert_document_to_entities
from .base import BaseTrainer
logger = logging.getLogger('mlflow')


class MentionDetectionTrainer(BaseTrainer):
    """Trainer for BIO sequence labeling mention detection
    """

    def __init__(self, config: dict):
        super().__init__(config)
        md_config = config['mention_detection']

        self.model = self.accelerator.prepare(
            MdBioModel(md_config['plm_name']))
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler.LambdaLR = None
        self.loss_fn: nn.CrossEntropyLoss = None
        self.train_dataset: MentionDetectionDataset = None
        self.test_dataset: MentionDetectionDataset = None

    def load_data(self, training: bool = True):
        data_config = self.config['data']
        md_config = self.config['mention_detection']
        tokenizer = md_config['plm_name']
        max_len = md_config['max_len']

        if training:
            train_dataset = from_owner(
                data_config['train_dataset_path'],
                data_config['train_dataset_name']
            )
            mlflow.log_input(train_dataset, context='md_train')
            self.train_dataset = MentionDetectionDataset(
                train_dataset.dataset, tokenizer, max_len)

        test_dataset = from_owner(
            data_config['test_dataset_path'],
            data_config['test_dataset_name']
        )
        mlflow.log_input(test_dataset, context='md_test')
        self.test_dataset = MentionDetectionDataset(
            test_dataset.dataset, tokenizer, max_len)

    def train(self):
        logger.info("Training Mention Detection")
        md_config = self.config['mention_detection']
        num_epochs = md_config['num_epochs']
        batch_size = md_config['batch_size']
        num_workers = get_num_workers()

        train_dataloader = self.accelerator.prepare(DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers))
        test_dataloader = self.accelerator.prepare(DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers))
        num_training_steps = len(train_dataloader) * num_epochs

        self.optimizer = self.accelerator.prepare(optim.Adam(
            self.model.parameters(), md_config['learning_rate']))
        self.scheduler = self.accelerator.prepare(get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps))
        self.loss_fn = self.accelerator.prepare(nn.CrossEntropyLoss(
            ignore_index=IGNORE_VALUE))

        for epoch in range(1, num_epochs+1):
            logger.info('Epoch %s/%s', epoch, num_epochs)
            self.train_one_epoch(epoch, train_dataloader, test_dataloader)

    def evaluate(self):
        logger.info("Evaluating Mention Detection")
        md_config = self.config['mention_detection']
        batch_size = md_config['batch_size']
        num_workers = get_num_workers()

        test_dataloader = self.accelerator.prepare(DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers))
        self.evaluate_dataloader(
            test_dataloader, 'test')

    def train_one_epoch(self, epoch: int, train_dataloader: DataLoader, test_dataloader: DataLoader):
        """Train model for one epoch
        Args:
            epoch (int): current epoch
        """
        train_dataloader_len = len(train_dataloader)

        # Train over epoch
        self.model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            current_step = (epoch-1) * train_dataloader_len + i

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            pred_logits = self.model(input_ids, attention_mask)

            self.optimizer.zero_grad()
            loss = self.loss_fn.forward(
                pred_logits.permute(0, 2, 1), labels)

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

            # Log metrics

            mlflow.log_metrics({
                'md_train_batch_cross-entropy': loss.item(),
            }, step=current_step)
        self.model.eval()

        logging.info('Validation')
        self.evaluate_dataloader(
            train_dataloader, step=current_step, context='train_epoch')
        self.evaluate_dataloader(
            test_dataloader, step=current_step, context='test_epoch')

    def evaluate_dataloader(self, dataloader: DataLoader, context: str, step: int = None):
        """Run evaluation for a dataloader
        Args:
            dataloader (DataLoader): dataloader to evaluate
            context (str): context of evaluation (train, dev, test, ...)
            step (int, optional): step to log. Defaults to None.
        """
        dataset: Dataset = cast(MentionDetectionDataset,
                                dataloader.dataset).dataset

        y_true: Dict[int, MiniDocument] = {}
        y_pred: Dict[int, MiniDocument] = {}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                ignored = batch['ignored']
                word_ids = batch['word_ids']
                documents_idx = batch['document_idx'].tolist()
                sentences_idx = batch['sentence_idx'].tolist()
                offsets = batch['offset'].tolist()

                pred_logits = self.model(input_ids, attention_mask)
                pred_labels = pred_logits.argmax(dim=2)

                # Need to mask all ignored indexes in order to compute entity labels
                pred_labels = pred_labels.clone()
                pred_labels[ignored] = IGNORE_VALUE

                for i_batch, document_idx in enumerate(documents_idx):
                    if document_idx not in y_true:
                        y_true[document_idx] = convert_document_to_entities(
                            dataset.documents[document_idx])

                    y_pred_entities = convert_bio_to_entities(
                        pred_labels[i_batch], word_ids[i_batch], sentences_idx[i_batch],
                        offsets[i_batch], dataset.documents[document_idx].id)
                    if document_idx not in y_pred:
                        y_pred[document_idx] = y_pred_entities
                    else:
                        y_pred[document_idx].entities.extend(
                            y_pred_entities.entities)

            y_true = [y_true[k] for k in sorted(y_true)]
            y_pred = [y_pred[k] for k in sorted(y_pred)]

            evaluate_mention_detection(
                y_true, y_pred, dataset.metadata.entity_types, context, step, draw_cm=len(dataset.metadata.entity_types) < 1000)

    def predict_dataset(self, dataset: Dataset) -> List[MiniDocument]:
        """Predict dataset
        Args:
            dataset (MentionDetectionDataset): dataset
        Returns:
            List[MiniDocument]: predicted entities
        """
        md_config = self.config['mention_detection']
        batch_size = md_config['batch_size']
        num_workers = get_num_workers()
        tokenizer = md_config['plm_name']
        max_len = md_config['max_len']
        dataloader = self.accelerator.prepare(DataLoader(
            MentionDetectionDataset(dataset, tokenizer, max_len),
            batch_size=batch_size, num_workers=num_workers))

        y_pred: Dict[int, MiniDocument] = {}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                ignored = batch['ignored']
                word_ids = batch['word_ids']
                documents_idx = batch['document_idx'].tolist()
                sentences_idx = batch['sentence_idx'].tolist()
                offsets = batch['offset'].tolist()

                pred_logits = self.model(input_ids, attention_mask)
                pred_labels = pred_logits.argmax(dim=2)

                # Need to mask all ignored indexes in order to compute entity labels
                pred_labels = pred_labels.clone()
                pred_labels[ignored] = IGNORE_VALUE

                for i_batch, document_idx in enumerate(documents_idx):
                    y_pred_entities = convert_bio_to_entities(
                        pred_labels[i_batch], word_ids[i_batch], sentences_idx[i_batch],
                        offsets[i_batch], dataset.documents[document_idx].id)
                    if document_idx not in y_pred:
                        y_pred[document_idx] = y_pred_entities
                    else:
                        y_pred[document_idx].entities.extend(
                            y_pred_entities.entities)

            y_pred = [y_pred[k] for k in sorted(y_pred)]
        return y_pred

    def save_model(self, folder: str):
        super().save_model(folder)
        torch.save(self.model.state_dict(), f'{folder}/mention_detection.pt')

    def load_model(self, folder: str):
        self.model.load_state_dict(torch.load(
            f'{folder}/mention_detection.pt'))
        self.model.eval()
