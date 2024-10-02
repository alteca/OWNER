"""Model for evaluation
"""
from typing import List
from ..data.model import Dataset, Document, Entity, MiniDocument


def convert_document_to_entities(document: Document) -> MiniDocument:
    """Convert dataset document to list of entities
    Args:
        document (Document): document
    Returns:
        MiniDocument: document
    """
    entities = [Entity(sentence_idx=entity.sentence_idx,
                       start_word_idx=entity.start_word_idx,
                       end_word_idx=entity.end_word_idx,
                       type=entity.type)
                for entity in document.entities]
    return MiniDocument(id=document.id, entities=entities)


def merge_dataset_with_predictions(dataset: Dataset, predictions: List[MiniDocument]) -> Dataset:
    """Merge dataset with predictions
    Args:
        dataset (Dataset): dataset
        predictions (List[MiniDocument]): predictions
    Returns:
        Dataset: dataset with predictions (in place of ground truth)
    """
    dataset = dataset.model_copy(deep=True)

    assert len(dataset.documents) == len(predictions)

    # Order predictions
    predictions_sorted = {
        prediction.id: prediction for prediction in predictions}

    entity_types = set()
    for document in dataset.documents:
        entities = []

        pred_document = predictions_sorted.get(document.id, None)
        if pred_document is not None:
            for entity in pred_document.entities:
                entity_types.add(entity.type)
                entities.append(Entity(
                    type=str(entity.type),
                    sentence_idx=entity.sentence_idx,
                    start_word_idx=entity.start_word_idx,
                    end_word_idx=entity.end_word_idx,
                ))
        document.entities = entities
    dataset.metadata.entity_types = entity_types
    return dataset
