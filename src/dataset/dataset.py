import numpy as np
from typing import Union, List, Dict, Tuple
from pathlib import Path
import logging


class ChronoMoviesDataset:
    """Dataset class to output movies and ranking for user, chronologically.
    The dataset will index all possible inputs using sliding window approach (where applicable)."""
    def __init__(self, mode: str, training_configuration):
        self.cfg = training_configuration
        self.raw_data: np.ndarray = []
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.names: Dict[str, int] = None
        self.split: np.ndarray = None
        self.unique_movies: List[int] = None
        self.unique_users: List[int] = None

    def load_ratings(self, ratings_path: Union[Path, str]):
        """Load csv file with ratings data."""
        with open(ratings_path, encoding="UTF-8") as ratings_file:
            self.names = {key.split("\n")[0]: i for i, key in enumerate(ratings_file.readline().split(","))}
            for line in ratings_file.readlines():
                self.raw_data.append(line.split(','))

        self.raw_data = np.asarray(self.raw_data).astype(float)
        self.get_split()
        self.logger.info(f"Loaded {len(self.raw_data)} of raw entries"
                         f"With columns named: {self.names}"
                         f"Mode {self.mode} consists of {len(self.split)} entries")
        self.unique_users = np.unique(self.split[:, self.names["userId"]])
        self.unique_movies = np.unique(self.split[:, self.names["movieId"]])

    def get_split(self, ratio: float = 0.9):
        """Get train and evaluation splits based on timestamp information."""
        self.raw_data = self.raw_data[self.raw_data[:, self.names["timestamp"]].argsort()]
        if self.mode == "train":
            self.split = self.raw_data[:int(ratio * len(self.raw_data))]
            np.random.shuffle(self.split)
        else:
            self.split = self.raw_data[-int((1 - ratio) * len(self.raw_data)):]

    def get_user(self, user_id: int) -> np.ndarray:
        """Get data for specific user."""
        user_data = self.raw_data[self.raw_data[:, self.names["userId"]] == user_id]
        return user_data

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        return {"user": self.split[idx, self.names["userId"]],
                "movie": self.split[idx, self.names["movieId"]],
                "rating": self.split[idx, self.names["rating"]]}

def main():
    d = ChronoMoviesDataset("train", None)
    d.load_ratings("data/ratings.csv")
    d.get_user(3)
    print(d.names)
    print(d[3])
    # a, b = d.get_splits()
    # print(a.shape)
    # print(b.shape)
    pass


if __name__ == "__main__":
    main()
