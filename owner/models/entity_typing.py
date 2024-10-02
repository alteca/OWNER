"""Models for Entity Typing
"""
import torch
from torch import nn
import numpy as np
from transformers import AutoModel
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV


class EntityEncodingModel(nn.Module):
    """Entity Typing encoder (also used during embedding refinement)
    """

    def __init__(self, plm_name: str):
        """Constructor
        Args:
            plm_name (str): bert model to use
        """
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name, return_dict=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                mask_index: torch.Tensor) -> torch.Tensor:
        """Forward function for model
        Args:
            input_ids (torch.Tensor): ids of tokens. Shape [batch, max_len]
            attention_mask (torch.Tensor): attention mask. Shape [batch, max_len]
            mask_index (torch.Tensor): Index of the [MASK] token for each instance 
                in the batch. Shape [batch]
        Returns:
            torch.Tensor: entity embeddings. Shape [batch, 768]
        """
        token_embeddings = self.plm(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        entity_embedding = token_embeddings[torch.arange(
            token_embeddings.shape[0]), mask_index.flatten()].view(token_embeddings.shape[0], -1)
        entity_embedding = self.dropout(entity_embedding)
        return entity_embedding


def bic_kmeans(estimator: KMeans, X: np.ndarray) -> float:
    """
    Calculate the Bayesian Information Criterion (BIC) for a KMeans result.
    The formula is using the BIC calculation for the Gaussian special case.

    Taken from: https://github.com/TankredO/pyckmeans/blob/main/pyckmeans/core/ckmeans.py

    Args:
        estimator (KMeans): kmeans model
        X (np.ndarray):  n * m matrix, where n is the number of samples (observations) and m is
        the number of features (predictors).

    Returns:
        float: BIC
    """

    k = estimator.cluster_centers_.shape[0]
    n = X.shape[0]

    return n * np.log(estimator.inertia_/n) + np.log(n) * k


def bic_scorer(*args) -> float:
    """-BIC
    GridSearchCV support only maximizing a metric (whereas we need to minimize BIC)
    Returns:
        float: -BIC
    """
    return - bic_kmeans(*args)


class AutoKmeans():
    """Kmeans clustering that estimates the number of clusters.
    """

    def __init__(self, k_min: int, k_max: int, k_step: int = 1, seed: int = None):
        """Constructor
        Args:
            k_min (int): minimum number of clusters
            k_max (int): upper bound for the number of clusters
            k_step (int): step for k identification
            seed (int, optional): random seed. Defaults to None.
        """
        # Template Kmeans
        base_kmeans = KMeans(
            init="k-means++", n_init=10, random_state=seed)

        # Grid search to find k (optimal number of clusters)
        self._kmeans = GridSearchCV(base_kmeans,
                                    param_grid={
                                        "n_clusters": range(k_min, k_max+1, k_step)
                                    },
                                    scoring={"bic": bic_scorer},
                                    verbose=2, n_jobs=1,
                                    refit="bic",
                                    )

    @property
    def k(self) -> int:
        """Return number of clusters
        Returns:
            int: number of clusters
        """
        return self._kmeans.best_params_['n_clusters']

    def get_scores(self) -> tuple:
        """Return grid search results
        Returns:
            tuple: grid search results
        """
        num_clusters = self._kmeans.cv_results_['param_n_clusters']
        bic_scores = - self._kmeans.cv_results_['mean_test_bic']

        return num_clusters, bic_scores

    def fit(self, entity_embeddings: torch.tensor):
        """Fit Kmeans
        Args:
            entity_embeddings (torch.Tensor): entity embeddings
        """
        entity_embeddings = entity_embeddings.cpu()

        # Run grid search
        self._kmeans.fit(entity_embeddings)

    def predict(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict clustering
        Args:
            entity_embeddings (torch.Tensor): entity_embeddings
        Returns:
            torch.Tensor: predicted clusters
        """
        return torch.from_numpy(self._kmeans.predict(entity_embeddings))
