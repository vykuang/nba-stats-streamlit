# Modelling NBA career stats with clustering

Having trained the model with `train.py` and logged to mlflow, I need to use the model and label the training dataset (season 2021-22) so that I know which player belongs to which labels, and ultimately if those labels make sense.

## MLflow

Stick to local backend store (mlflow.db) and artifact root (./mlruns). Don't really need to setup a new instance.

Could set up MLflow as a separate docker service however, and dockerize the DB as another service, with the artifact root as a volume when it goes to production.

### Logging

Discussed in `models.ipynb`, but one thing to note for convenience is `n_labels`. That should also be logged as a parameter

- kmeans: `.labels_`
- hierarchical: `.labels_`
- dbscan: `.labels_`; noisy labels (not belonging to any cluster) are assigned `-1`.

### Registry

Criteria for model registry:

- `n_labels` >= 3: any fewer would be an underfit
- Highest `silhouette_score`

## Visualizing the learned labels

How to go about this. Seems like a manual process. Ah Lebron is group 0, and Bam is in 1? Does that make sense?

Given the most successful model I need to find the centroids and the corresponding player_id and names. Trick is that not all models provide that so readily.

How about this. Given a label, I filter through all the players belong to that label, and return the players with the most gaudy statlines in terms of `PTS` + `REB` + `AST`, and use that to make a human decision on which label means what.
