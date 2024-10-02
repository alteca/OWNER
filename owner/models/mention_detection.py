"""Entity Extraction model
"""
import torch
from torch import nn
from transformers import AutoModel


class MdBioModel(nn.Module):
    """Model for BIO tagging
    """

    def __init__(self, plm_name: str):
        """Constructor
        Args:
            plm_name (str): bert model to use
        """
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name, return_dict=True)
        self.classifier = nn.Sequential(*[
            nn.Dropout(0.1),
            nn.Linear(self.plm.config.hidden_size, 3)
        ])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward function for model
        Args:
            input_ids (torch.Tensor): ids of tokens. Shape [batch, max_len]
            attention_mask (torch.Tensor): attention mask. Shape [batch, max_len]
        Returns:
            torch.Tensor: predicted logits for each token and each class.
                Shape [batch, max_len, 3]
        """
        token_embeddings = self.plm(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        token_logits = self.classifier(token_embeddings)
        return token_logits
