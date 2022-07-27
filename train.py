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
    # prepare_folders(checkpoints_dir=train_cfg.CHECKPOINTS_DIR)
    logger = create_logger(train_cfg.LOGS_DIR, train_cfg.NAME, verbose_level=verbose_level)
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        mlflow.log_artifact("config/training_config.py")
        logger.info(f"Training datapath: {train_cfg.RATINGS_PATH}")

        train_dataset = ChronoMoviesDataset(mode="train", training_configuration=train_cfg)
        eval_dataset = ChronoMoviesDataset(mode="eval", training_configuration=train_cfg)

        train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True, drop_last=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=False, drop_last=True)
        dataloaders = {"train": train_dataloader, "eval": eval_dataloader}
        print(next(iter(train_dataloader)))

        movie_model = model_helper["movie_model"]
        user_model = model_helper["user_model"]
        trainer = Trainer(movie_l=train_dataset.movie_list, user_l=train_dataset.user_list, dataloaders=dataloaders, loss=model_helper["retrieval_loss"], user_model=user_model, movie_model=movie_model, training_config=train_cfg)
        trainer()

        torch.save(trainer.movie_model, "movie_model.pt")
        torch.save(trainer.user_model, "user_model.pt")
        exit()
        # train_features, train_labels = train_dataloader["features"], train_dataloader["labels"]

        # logger.info(f"Labels shape: {train_labels.shape}")
        # logger.info(f"Features shape: {train_features.shape}")

        exit()
        if train_cfg.MODEL_PATH is not None:
            model = joblib.load(train_cfg.MODEL_PATH)
            logger.info(f"Successfully loaded model from {train_cfg.MODEL_PATH}")
        else:
            model = model_helper[train_cfg.MODEL_TO_USE]
        logger.info(model)

        with open(train_cfg.EMOTIONS_FILE, encoding="UTF-8") as emotions_file:
            emotions = [emotion.split(",") for emotion in emotions_file.readlines()][0]
        model.classes_ = emotions
        model.class_weight = class_weights

        trainer = Trainer(dataloaders, model, train_cfg)
        trainer()

        # save model
        joblib.dump(trainer.model,
                    train_cfg.CHECKPOINTS_DIR / Path(f"{train_cfg.MODEL_TO_USE}_model_{train_cfg.MAX_ITER}.joblib"))
        mlflow.log_artifact(train_cfg.CHECKPOINTS_DIR / Path(f"{train_cfg.MODEL_TO_USE}_model_{train_cfg.MAX_ITER}.joblib"))
        mlflow.sklearn.log_model(model, "model_mlflow")

        # get metrics to compare trainings
        mlflow.sklearn.eval_and_log_metrics(
            model,
            train_dataloader["features"],
            train_dataloader["labels"],
            prefix="training_")
        mlflow.sklearn.eval_and_log_metrics(
            model,
            eval_dataloader["features"],
            eval_dataloader["labels"],
            prefix="val_")
        logger.info("Optimization finished.")


if __name__ == "__main__":
    main()
