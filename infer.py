import numpy as np
import pandas as pd
from src.models.retrieval_loss import RetrievalLoss
import torch
from torch.utils.data import DataLoader
import argparse
from src.dataset.dataset import ChronoMoviesDataset
from config.training_config import get_training_config
from pathlib import Path

import joblib
import mlflow

from config.training_config import get_training_config
from src.models.model_helper import model_helper
from src.trainer import Trainer
from src.utils.logger import create_logger
# from src.utils.prepare_folders import prepare_folders


def main():
    parser = argparse.ArgumentParser(description="Emotion classifier training.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    verbose_level: str = args.verbose_level

    train_cfg = get_training_config()
    logger = create_logger(train_cfg.LOGS_DIR, train_cfg.NAME, verbose_level=verbose_level)
    mlflow.sklearn.autolog()
    with mlflow.start_run():

        train_dataset = ChronoMoviesDataset(mode="train", training_configuration=train_cfg)
        movies = train_dataset.movie_list
        movies_df = pd.read_csv("data/movies.csv")[:200]
        for i in np.arange(1, 99, 17):
            user = torch.tensor([i])
            movies = torch.tensor(movies).long()

            movie_model = torch.load("movie_model.pt")
            user_model = torch.load("user_model.pt")

            user_embedding = user_model(user)
            movies_embedding = movie_model(movies)
            print(user_embedding)
            print(movies_embedding)
            recommendations = torch.argsort(torch.nn.functional.cosine_similarity(user_embedding, movies_embedding)).detach().cpu().numpy()
            print(recommendations[:10])
            print(movies_df.iloc[recommendations[:3]])
            print(i)
            watched = train_dataset.raw_data[train_dataset.raw_data[:, train_dataset.names["userId"]] == i][:, train_dataset.names["movieId"]]
            # print(watched)
            print(f"Movies that the user watched {movies_df.iloc[watched]}")

if __name__ == "__main__":
    main()
