from config.training_config import get_training_config
from torchtext.vocab import vocab
from collections import OrderedDict
import torch
import numpy as np


class MovieModel(torch.nn.Module):
    def __init__(self, training_configuration, movie_ids):
        super().__init__()

        # print(len(movie_ids))
        num_embeddings = np.arange(sorted(movie_ids)[-1])
        print(num_embeddings)
        self.embedding = torch.nn.Embedding(num_embeddings=len(num_embeddings) + 6,
                                            embedding_dim=training_configuration.EMBEDDING_DIM)
        print(self.embedding)

    def forward(self, x):
        # print(x)
        x = self.embedding(x)

        return x

