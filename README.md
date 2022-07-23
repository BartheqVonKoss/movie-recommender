# movie-recommender
### General description
### Overall scheme
Insert a picture (or more) here.
### Dataset
The dataset used consists of ratings of movies over some period of time.
#### Discussion
As per my current understanding, it might be hard to judge what movie(s) to recommend. We'd need a surrogate model for that. For the surrogate model, the goal is to reliably predict the next movie a user will watch. For that the input should be ordered chronologically. We can use slidint window apporoach, to get more data out of what we have. This will still be less than one would have if the choice of treating the dataset in a time series manner was not made. 
I expect the dataset being not too hard to get reasonable result but the main target of mine is to build a system around it.
#### Obtain the dataset
The dataset can be obtained at [this address](https://grouplens.org/datasets/movielens/). One should unzip it and place in the prepared directory `data`. This is not trully necessary because of the usage of configuration files that keep track of the used paths, yet it is recommended by the author.  
#### Credits
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>. 
Much appreciated the effort for combining it.

### Model
The model developed here aims to select a good movie recommendation to the user. The task is accomplished in two stages:
- retrieval of relevant movies to watch
- rank retrieved movies by relevance for the user.

### Installation

#### Dependencies
The project has been developed on a linux machine and hence the reproduction steps for it will be laid. There is an idea of making the system run os-agnostically but I will leave it for more mature times (of this project).

### Miscellaneous
Code should follow pep8 convention. Development setup of that from `setup.cfg` should be followed to contribute to this repository. I do not claim author rights at any given point of time. 
