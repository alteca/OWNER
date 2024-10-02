"""Scorer for mention detection task
"""
from typing import List, Optional
import json
import mlflow
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from ..data.model import MiniDataset, MiniDocument, Metadata, Entity
from ..data.serialization import serialize_mini_owner_dataset


def convert_bio_to_entities(bio_tags: torch.Tensor, word_ids: torch.Tensor,
                            sentence_idx: int, offset: int,
                            document_id: str) -> MiniDocument:
    """Convert bio tags to entity entities format
    Args:
        bio_tags (torch.Tensor): bio tags.
            Values: 0=o, 1=b, 2=i (int tensor)
            Shape: [length]
        word_ids: (torch.Tensor): convert tokens to words
        sentence_idx (int): sentence index
        offset (int): offset for entity indices
    Returns:
        EvalDocument: document
    """
    bio_tags = bio_tags.cpu()
    word_ids = word_ids.cpu()

    assert len(bio_tags.shape) == 1
    assert not bio_tags.is_floating_point()

    entities: List[Entity] = []

    state = 0  # 0=outside, 1=inside an entity
    start_pos = 0
    end_pos = 0
    for i, tag in enumerate(bio_tags):
        match state:
            case 0:
                match tag:
                    case 0:
                        continue
                    case 1:  # New entity has started
                        state = 1
                        start_pos = i
                        end_pos = i
                    case 2:  # Not possible / Invalid
                        continue
            case 1:
                match tag:
                    case 0:  # Entity has ended
                        state = 0
                        entities.append(Entity(sentence_idx=sentence_idx,
                                               start_word_idx=start_pos,
                                               end_word_idx=end_pos,
                                               type="entity"))
                    case 1:  # New entity has started
                        state = 1
                        entities.append(Entity(sentence_idx=sentence_idx,
                                               start_word_idx=start_pos,
                                               end_word_idx=end_pos,
                                               type="entity"))
                        start_pos = i
                    case 2:  # Entity is continuing
                        end_pos = i
                        continue
    if state == 1:
        entities.append(Entity(sentence_idx=sentence_idx,
                               start_word_idx=start_pos,
                               end_word_idx=end_pos, type="entity"))
    # Convert token index to word index
    for entity in entities:
        entity.start_word_idx = word_ids[entity.start_word_idx].item() + offset
        entity.end_word_idx = word_ids[entity.end_word_idx].item() + offset + 1

    return MiniDocument(id=document_id, entities=entities)


def evaluate_mention_detection(true_mentions: List[MiniDocument], pred_mentions: List[MiniDocument],
                               true_entity_types: List[str | int], context: str,
                               step: Optional[int] = None, draw_cm: bool = True):
    """Evaluation for mention detection

    Args:
        true_mentions (List[MiniDocument]): true mentions
        pred_mentions (List[MiniDocument]): predicted mentions
        true_entity_types (List[str | int]): list of entity types of true mentions
        context (str): context (for mlflow logging)
        step (Optional[int], optional): step of evaluation (for mlflow logging). Defaults to None.
    """
    cm_context = f'{context}{f"_{step}" if step is not None else ""}'
    true_entity_types = [str(t) for t in sorted(true_entity_types)]

    # Compute confusion matrix
    tp = {t: 0 for t in true_entity_types}
    # tn = 0 Always 0
    fn = {t: 0 for t in true_entity_types}
    fp = 0

    true_mentions = {document.id: document for document in true_mentions}
    pred_mentions = {document.id: document for document in pred_mentions}
    document_ids = set(list(true_mentions.keys())).union(
        list(pred_mentions.keys()))

    serialize_mini_owner_dataset(MiniDataset(documents=pred_mentions.values(), metadata=Metadata(
        entity_types=set(['entity']))), f'md_{cm_context}_pred_dataset.json')
    mlflow.log_artifact(f'md_{cm_context}_pred_dataset.json')

    for document_id in document_ids:
        true_document = true_mentions.get(
            document_id, MiniDocument(id=document_id, entities=[]))
        pred_document = pred_mentions.get(
            document_id, MiniDocument(id=document_id, entities=[]))

        # Keep track of matched entities
        true_matched = torch.zeros([len(true_document.entities)])
        pred_matched = torch.zeros([len(pred_document.entities)])

        # True positives
        for i_true, true_entity in enumerate(true_document.entities):
            for i_pred, pred_entity in enumerate(pred_document.entities):
                if true_entity.sentence_idx == pred_entity.sentence_idx and \
                        true_entity.start_word_idx == pred_entity.start_word_idx and \
                        true_entity.end_word_idx == pred_entity.end_word_idx:
                    if true_matched[i_true] == 0 and pred_matched[i_pred] == 0:
                        # To avoid adding many TP
                        tp[true_entity.type] += 1
                    true_matched[i_true] = 1
                    pred_matched[i_pred] = 1

        # False negatives
        for i_true, matched in enumerate(true_matched):
            if matched == 0:
                fn[true_document.entities[i_true].type] += 1

        # False positives
        fp += len(pred_matched) - pred_matched.sum().item()

    confusion_matrix = torch.zeros(
        [len(true_entity_types) + 1, 2], dtype=torch.int32)
    for i_type, entity_type in enumerate(sorted(true_entity_types)):
        confusion_matrix[i_type, 0] = tp[entity_type]
        confusion_matrix[i_type, 1] = fn[entity_type]
    confusion_matrix[-1, 0] = fp

    # Aggregated scores
    tp = confusion_matrix[:-1, 0].sum()
    pp = confusion_matrix[:, 0].sum().float()
    p = confusion_matrix[:-1, :].sum().float()

    recall = tp / p if p > 0 else 0.
    precision = tp / pp if pp > 0 else 0.
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (recall > 0 or precision > 0) else 0.

    mlflow.log_metrics({
        f'md_{context}_recall': recall,
        f'md_{context}_precision': precision,
        f'md_{context}_f1': f1,
    }, step=step)

    # Confusion matrix
    if draw_cm:
        true_labels = true_entity_types + ['∅']
        pred_labels = ['entity', '∅']

        fig = plt.figure(figsize=(len(pred_labels) * 0.67 + 4,
                                  len(true_labels) * 0.67))
        sns.heatmap(confusion_matrix, annot=True, fmt='d',
                    xticklabels=pred_labels, yticklabels=true_labels)
        plt.xlabel("Predicted entity")
        plt.ylabel("True entity")
        fig.tight_layout()
        mlflow.log_figure(fig, f'md_{cm_context}_confusion-matrix.png')
        plt.close()

        cm_path = f'md_{cm_context}_confusion-matrix.pt'
        cm_metadata_path = f'md_{cm_context}_confusion-matrix-metadata.json'
        torch.save(confusion_matrix, cm_path)
        with open(cm_metadata_path, 'w', encoding='utf-8') as file:
            json.dump({'pred_labels': pred_labels,
                       'true_labels': true_labels}, file)
        mlflow.log_artifact(cm_metadata_path)
        mlflow.log_artifact(cm_path)
