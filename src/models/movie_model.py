from sklearn.preprocessing import LabelEncoder
from config.training_config import get_training_config
from torchtext.vocab import vocab
from collections import OrderedDict
import torch
import numpy as np


class MovieModel(torch.nn.Module):
    def __init__(self, training_configuration, movie_ids):
        super().__init__()

        self.label_encoder = LabelEncoder().fit(movie_ids)
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.label_encoder.transform(movie_ids)),
                                            embedding_dim=training_configuration.EMBEDDING_DIM)

    def forward(self, x):
        x = torch.from_numpy( self.label_encoder.transform(x) )
        x = self.embedding(x)

        return x

