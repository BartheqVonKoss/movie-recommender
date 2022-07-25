import numpy as np
from config.training_config import get_training_config
from torchtext.vocab import vocab
from collections import OrderedDict
import torch

# user_model = tf.keras.Sequential([
#   tf.keras.layers.StringLookup(
#       vocabulary=unique_user_ids, mask_token=None),
#   # We add an additional embedding to account for unknown tokens.
#   tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
# ])


class UserModel(torch.nn.Module):
    def __init__(self, training_configuration, user_ids):
        super().__init__()

        print(len(user_ids) + 1)

        num_embeddings = len( np.arange(sorted(user_ids)[-1]) )
        print(num_embeddings)
        print(training_configuration.EMBEDDING_DIM)
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings + 5,
                                            embedding_dim=training_configuration.EMBEDDING_DIM)
        print(self.embedding)

    def forward(self, x):
        # print(x, x.shape)
        x = self.embedding(x)
        # print(x, x.shape)
        # exit()

        return x


def main():
    import numpy as np
    m = UserModel(get_training_config(), np.arange(501))
    print(m)
    # import numpy as np 
    # a = np.arange(501)
    # b = np.array([10, 12, 16])
    samples = torch.tensor( [[10, 12, 1], [22, 21, 11]] )
    print(m(samples))
    print(samples.shape)
    print(m(samples).shape)



if __name__ == "__main__":
    main()
