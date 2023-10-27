import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer


class DotSimilarityLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss()):
        super(DotSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = torch.sum(torch.mul(embeddings[0], embeddings[1]), dim=1)
        return self.loss_fct(output, labels.view(-1))