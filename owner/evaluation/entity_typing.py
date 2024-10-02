"""Scorer for entity typing (and NER) task
"""
from typing import List, Optional, Literal
import json
import mlflow
import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score
from sklearn.preprocessing import normalize as sklearn_normalize
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from ..data.model import MiniDataset, MiniDocument, Metadata
from ..data.serialization import serialize_mini_owner_dataset


def clustering_confusion_matrix(true_entities: np.ndarray, pred_entities: np.ndarray,
                                true_entity_types: List[str | int],
                                pred_entity_types: List[str | int],
                                normalize: Literal["true", "pred", "all", None] = None) -> tuple:
    """Compute clustering confusion matrix. See https://stackoverflow.com/questions/55764091
    Args:
        true_entities (np.ndarray): true labels (int)
        pred_entities (np.ndarray): predicted clusters (int)
        true_entity_types: (List[str | int]): true entity types
        pred_entity_types: (List[str | int]): pred entity types
        normalize (string): see 
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    Returns:
        tuple: confusion matrix, true labels (y), predicted clusters labels (x)
    """
    # if y_true and y_pred contain missing entities (labeled with -1)
    missing = -1 in true_entities or -1 in pred_entities

    # Compute confusion matrix
    cmatrix = confusion_matrix(true_entities, pred_entities)
    y_true_labels = np.arange(cmatrix.shape[0])
    y_pred_labels = np.arange(cmatrix.shape[1])
    if missing:
        y_true_labels -= 1
        y_pred_labels -= 1

    # Compute an approximate alignment between true labels and clusters (to have a diagonal in the
    # confusion matrix)
    indexes = linear_sum_assignment(- cmatrix + np.max(cmatrix))
    cmatrix = cmatrix[:, indexes[1]]
    y_pred_labels = y_pred_labels[indexes[1]]

    # Sort confusion matrix to have a band matrix (to regroup confusions)
    graph = csr_matrix(cmatrix > 0.01 * cmatrix.sum())
    output = reverse_cuthill_mckee(graph, symmetric_mode=False)
    cmatrix = cmatrix[output][:, output]
    y_pred_labels = y_pred_labels[output]
    y_true_labels = y_true_labels[output]

    # Remove empty lines and columns
    y_true_size = len(true_entity_types)
    y_pred_size = len(pred_entity_types)
    y_pred_selector = y_pred_labels < y_pred_size
    y_pred_labels = y_pred_labels[y_pred_selector]
    y_true_selector = y_true_labels < y_true_size
    y_true_labels = y_true_labels[y_true_selector]
    cmatrix = cmatrix[y_true_selector][:, y_pred_selector]

    # Put missing row and column at the end of the matrix
    y_true_labels = y_true_labels.tolist()
    y_pred_labels = y_pred_labels.tolist()
    y_true_missing_index = y_true_labels.index(
        -1) if -1 in y_true_labels else None
    y_pred_missing_index = y_pred_labels.index(
        -1) if -1 in y_pred_labels else None

    if y_true_missing_index is not None:
        reorder_selector = [i for i in range(y_true_missing_index)] + [i+1 for i in range(
            y_true_missing_index, len(y_true_labels)-1)] + [y_true_missing_index]
        cmatrix = cmatrix[reorder_selector]
        y_true_labels.pop(y_true_missing_index)
        y_true_labels.append(-1)

    if y_pred_missing_index is not None:
        reorder_selector = [i for i in range(y_pred_missing_index)] + [i+1 for i in range(
            y_pred_missing_index, len(y_pred_labels)-1)] + [y_pred_missing_index]
        cmatrix = cmatrix[:, reorder_selector]
        y_pred_labels.pop(y_pred_missing_index)
        y_pred_labels.append(-1)

    # Normalize confusion matrix if needed
    if normalize == 'true':
        cmatrix = sklearn_normalize(cmatrix, norm='l1', axis=1)
    elif normalize == 'pred':
        cmatrix = sklearn_normalize(cmatrix, norm='l1', axis=0)
    elif normalize == 'all':
        cmatrix = cmatrix / cmatrix.sum().sum()

    return cmatrix, \
        y_true_labels, \
        y_pred_labels


def evaluate_entity_typing(true_entities: List[MiniDocument], pred_entities: List[MiniDocument],
                           true_entity_types: Optional[List[str | int]] = None,
                           pred_entity_types: Optional[List[str | int]] = None,
                           context: str = "",
                           step: Optional[int] = None, prefix: str = 'et'):
    """Scorer for entity typing

    Args:
        true_entities (List[MiniDocument]): true entities
        pred_entities (List[MiniDocument]): predicted entities
        true_entity_types (List[str | int]): list of entity types of true entities.
            Defaults to None.
        true_entity_types (List[str | int]): list of entity types of predicted entities.
        Defaults to None.
        context (str): context (for mlflow logging)
        step (Optional[int], optional): step of evaluation (for mlflow logging).
            Defaults to None.
        prefix (str). prefix for metric logging. Defaults to 'et'
    """
    cm_context = f'{context}{f"_{step}" if step is not None else ""}'

    if true_entity_types is None:
        true_entity_types = set()
        for document in true_entities:
            for entity in document.entities:
                true_entity_types.add(entity.type)
    true_entity_types = sorted(true_entity_types)
    true_entity_type_to_id = {t: i for i, t in enumerate(true_entity_types)}

    if pred_entity_types is None:
        pred_entity_types = set()
        for document in pred_entities:
            for entity in document.entities:
                pred_entity_types.add(entity.type)
    pred_entity_types = sorted(pred_entity_types)
    pred_entity_type_to_id = {t: i for i, t in enumerate(pred_entity_types)}

    serialize_mini_owner_dataset(MiniDataset(documents=pred_entities, metadata=Metadata(
        entity_types=set(pred_entity_types))), f'{prefix}_{cm_context}_pred_dataset.json')
    mlflow.log_artifact(f'{prefix}_{cm_context}_pred_dataset.json')

    true_entities = {document.id: document for document in true_entities}
    pred_entities = {document.id: document for document in pred_entities}
    document_ids = set(list(true_entities.keys())).union(
        list(pred_entities.keys()))

    # Internal representations for truth and predictions (true type and predicted cluster)
    y_true_internal: List[int] = []
    y_pred_internal: List[int] = []

    for document_id in document_ids:
        true_document = true_entities.get(
            document_id, MiniDocument(id=document_id, entities=[]))
        pred_document = pred_entities.get(
            document_id, MiniDocument(id=document_id, entities=[]))

        # Keep track of matched entities
        true_matched = np.zeros([len(true_document.entities)])
        pred_matched = np.zeros([len(pred_document.entities)])

        # True positives (and type confusions)
        for i_true, true_entity in enumerate(true_document.entities):
            for i_pred, pred_entity in enumerate(pred_document.entities):
                # Matches between a true entity and multiple predicted entities
                # or 1 predicted entity and multiple true entities is possible:
                # UniversalNER can predict multiple entity types for the same span
                if true_entity.sentence_idx == pred_entity.sentence_idx and \
                        true_entity.start_word_idx == pred_entity.start_word_idx and \
                        true_entity.end_word_idx == pred_entity.end_word_idx:
                    y_true_internal.append(
                        true_entity_type_to_id[true_entity.type])
                    y_pred_internal.append(
                        pred_entity_type_to_id[pred_entity.type])
                    true_matched[i_true] = 1
                    pred_matched[i_pred] = 1

        # False negatives
        for i_true, matched in enumerate(true_matched):
            if matched == 0:
                y_true_internal.append(
                    true_entity_type_to_id[true_document.entities[i_true].type])
                y_pred_internal.append(-1)  # No corresponding cluster

        # False positives
        for i_pred, matched in enumerate(pred_matched):
            if matched == 0:
                y_true_internal.append(-1)  # No corresponding type
                y_pred_internal.append(
                    pred_entity_type_to_id[pred_document.entities[i_pred].type])

    y_true_internal = torch.tensor(y_true_internal)
    y_pred_internal = torch.tensor(y_pred_internal)

    # Synthetic scores
    ami = adjusted_mutual_info_score(y_true_internal, y_pred_internal)

    mlflow.log_metrics({
        f'{prefix}_{context}_ami': ami,
    }, step=step)

    # Confusion matrix
    cmatrix, cmatrix_yticks, cmatrix_xticks = clustering_confusion_matrix(
        y_true_internal, y_pred_internal, true_entity_types, pred_entity_types)
    xticks_missing = cmatrix_xticks.index(-1) if -1 in cmatrix_xticks else None
    cmatrix_xticks = [pred_entity_types[t] for t in cmatrix_xticks]
    if xticks_missing is not None:
        cmatrix_xticks[xticks_missing] = '∅'
    yticks_missing = cmatrix_yticks.index(-1) if -1 in cmatrix_yticks else None
    cmatrix_yticks = [true_entity_types[t] for t in cmatrix_yticks]
    if yticks_missing is not None:
        cmatrix_yticks[yticks_missing] = '∅'

    try:    # Sometimes figures are too large to be printed
        fig = plt.figure(figsize=(len(cmatrix_xticks), len(cmatrix_yticks)))
        sns.heatmap(cmatrix, annot=True, xticklabels=cmatrix_xticks,
                    yticklabels=cmatrix_yticks, fmt='d')
        plt.ylabel("Truth")
        plt.xlabel('Predicted')
        plt.tight_layout()
        mlflow.log_figure(
            fig, f'{prefix}_{cm_context}_confusion-matrix.png')
        plt.close(fig)
    except:
        pass

    cm_path = f'{prefix}_{cm_context}_confusion-matrix.pt'
    cm_metadata_path = f'{prefix}_{cm_context}_confusion-matrix-metadata.json'
    torch.save(cmatrix, cm_path)
    with open(cm_metadata_path, 'w', encoding='utf-8') as file:
        json.dump({'xticklabels': cmatrix_xticks,
                  'yticklabels': cmatrix_yticks}, file)
    mlflow.log_artifact(cm_metadata_path)
    mlflow.log_artifact(cm_path)
