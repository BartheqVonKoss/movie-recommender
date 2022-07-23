# movie-recommender

### Dataset
The dataset used consists of ratings of movies over some period of time.
#### Discussion
As per my current understanding, it might be hard to judge what movie(s) to recommend. We'd need a surrogate model for that. For the surrogate model, the goal is to reliably predict the next movie a user will watch. For that the input should be ordered chronologically. We can use slidint window apporoach, to get more data out of what we have. This will still be less than one would have if the choice of treating the dataset in a time series manner was not made. 
#### Credits
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>. 
Much appreciated the effort for combining it.
