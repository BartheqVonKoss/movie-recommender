import numpy as np
import logging
from typing import Any, Dict

import numpy as np
from torch.utils.data import DataLoader
import torch



class Trainer:
    def __init__(self,
                 dataloaders: Dict[str, DataLoader],
                 user_model: torch.nn.Module,
                 movie_model: torch.nn.Module,
                 loss: torch.nn.Module,
                 user_l,
                 movie_l,
                 training_config):
        self.cfg = training_config
        self.dataloaders = dataloaders
        self.logger = logging.getLogger(__name__)
        self.loss = loss()
        self.param_distributions: Dict[str, Any] = None
        # self.get_parameters_space()

        # user = np.unique(next(iter(self.dataloaders["train"]))["user"])
        # movie = np.unique(next(iter(self.dataloaders["train"]))["movie"])
        self.user_model = user_model(self.cfg, user_l)
        self.movie_model = movie_model(self.cfg, movie_l)
        self.optimizer_u = torch.optim.Adam(self.user_model.parameters(), 1e-4)
        self.optimizer_m = torch.optim.Adam(self.movie_model.parameters(), 1e-4)

    def _epoch(self, mode: str = "train"):
        epoch_loss = 0
        for i, batch in enumerate(self.dataloaders[mode]):
            user_b = batch["user"].long()
            movie_b = batch["movie"].long()
            rating_b = batch["rating"]
            self.optimizer_m.zero_grad()
            self.optimizer_u.zero_grad()
            user_embeddings = self.user_model(user_b)
            movie_embeddings = self.movie_model(movie_b)
            loss = self.loss(user_embeddings, movie_embeddings)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer_m.step()
            self.optimizer_u.step()

        print(epoch_loss)

    def __call__(self) -> None:
        for epoch in range(self.cfg.MAX_EPOCHS):
            self._epoch("train")
        # X, y = self.datasets["train"]["features"], self.datasets["train"]["labels"]
        # optuna_search = optuna.integration.OptunaSearchCV(self.model, self.param_distributions)
        # get best parameters after cross-validation and hyperparameters search
        # optuna_search.fit(X, y)
        # set parameters to the model and train it
        # self.model.set_params(**optuna_search.best_params_)
        # self.model.fit(X, y)

    # def get_parameters_space(self) -> None:
    #     """Method to get proper parameters space given the model."""
    #     if isinstance(self.model, LogisticRegression):
    #         self.param_distributions = lr_param_distributions
    #     if isinstance(self.model, RandomForestClassifier):
    #         self.param_distributions = rf_param_distributions
