"""Holder of a dictionary that consists of pairs of name - model."""
from src.models.movie_model import MovieModel
from src.models.user_model import UserModel
from src.models.retrieval_loss import RetrievalLoss


model_helper = {
    "user_model": UserModel,
    "movie_model": MovieModel,
    "retrieval_loss": RetrievalLoss

}
