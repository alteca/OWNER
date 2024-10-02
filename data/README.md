# Dataset format

OWNER expects datasets to be in the OWNER format.

## Files

Each dataset is composed of three files:

- `train.json`. See format below.
- `test.json`.
- `dev.json`.

A dataset used for training must at least contain a `train.json` file, and a dataset used during evaluation must contain a `test.json` file.

## `train.json`, `dev.json`, `test.json` files format

The files are formatted as `json`. A file contains a json object with the following keys:

- `documents`: List of documents of the dataset. Each document is a json object with the following keys:
  - `id`: Unique id of the document.
  - `sentences`. List of the sentences of the document. Each sentence contains the list of tokens/words that composes it.
  - `entities`. List of entities. Each entity is a json object with the following keys:
    - `type`. Type of the entity.
    - `name`. Text of the mention.
    - `sentence_idx`. Index of the sentence where the mention is located.
    - `start_word_idx`. Index of the first token of the entity.
    - `end_word_idx`. Index of the last token of the entity (exclusive).
- `metadata`: Metadata of the file. It is a json object with the following keys:
  - `entity_types`: List of entity types appearing in the documents.

We provide scripts to read/write datasets in the OWNER format. See the files `owner/data/serialization.py` and `owner/data/model.py`.

### Sample document

```json
{
    "id": "1",
    "sentences": [
        [ "Skai", "TV", "is", "a", "Greek", "free", "-", "to", "-", "air", "television", "network", "based", "in", "Piraeus", "."],
        [ "It", "is", "part", "of", "the", "Skai", "Group", ",", "one", "of", "the", "largest", "media", "groups", "in", "the", "country", "."],
        ...
    ],
    "entities": [
        {
            "type": "ORG",
            "sentence_idx": 0,
            "start_word_idx": 0,
            "end_word_idx": 2
        },
        {
            "type": "ORG",
            "sentence_idx": 3,
            "start_word_idx": 18,
            "end_word_idx": 20
        },
        ...
    ],
}
```

### Sample metadata

```json
{
  "entity_types": ["ORG", "MISC", "PER", "LOC"]
}
```
