"""
Trains a selection of clusterer models on our compiled dataset,
and logs the results onto MLflow
"""
import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(data_path: Path) -> pd.DataFrame:
    """Loads the data file from intermediate storage"""
    df = pd.read_pickle(data_path)
    return df


def score_cluster(
    clusterer,
    data: pd.DataFrame,
    printout: bool = False,
) -> list[int]:
    """Evalutes cluster performance

    Parameters
    ----------
    clusterer
        fitted clusterer with `.labels_` attribute
    data
        A 2D numpy array or pandas dataframe of shape (M, N).

    Returns
    -------
    scores
        list of cluster evaluation scores (float)
    """
    try:
        silh = silhouette_score(data, labels=clusterer.labels_, metric="euclidean")
        ch = calinski_harabasz_score(data, labels=clusterer.labels_)
        db = davies_bouldin_score(data, labels=clusterer.labels_)
    except ValueError as e:
        print(e)
        print("Only one label found; setting silh = 0, ch = 0, db = 99")
        silh = 0
        ch = 0
        db = 99

    if printout:

        print(f"silhouette_score:\t\t{silh:.3f}")
        print(f"calinski_harabasz_score:\t{ch:.3f}")
        print(f"davies_bouldin_score:\t\t{db:.3f}")
    return silh, ch, db


def model_search(data: pd.DataFrame, num_trials: int = 10) -> Trials:
    """Trains the clusterers and search for the best model

    Parameters:
    -----------

    data
        DataFrame, or numpy 2D array with row-wise samples
    num_trials
        int
        Number of trials for hyperopt, for each model type

    Returns:
    --------

    trials
        hyperopt.Trials() object, essentially a dict containing the log metrics

    References:
    -----------

    Searching multiple models in hyperopt:
    https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-model-selection.html

    """
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXP_NAME = os.getenv("MLFLOW_EXP_NAME", "nba-player-cluster")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXP_NAME)

    def objective(params):
        """Used in conjunction with hyperopt.fmin"""
        with mlflow.start_run():
            # log only the hyperparameters passed
            mlflow.log_params(params)
            cls_type = params["type"]

            mlflow.set_tag("model", cls_type)
            # can't pass 'type' kwarg to our model; from databricks nb
            del params["type"]

            if cls_type == "kmeans":
                cls = cluster.KMeans(**params)
            elif cls_type == "hierarchical":
                cls = cluster.AgglomerativeClustering(**params)
            elif cls_type == "dbscan":
                cls = cluster.DBSCAN(**params)
            else:
                raise ValueError(
                    "Model type not accepted; use one of ['kmeans', 'hierarchical', 'dbscan']"
                )
            cluster_pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=4),
                cls,
            )

            # score using metrics that do not require ground truths
            cluster_pipe.fit(data)
            mlflow.log_artifact()
            silh, ch, db = score_cluster(cluster_pipe[-1], data)
            metrics_name = [
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
            ]
            # metrics = {name: score for name, score in zip(metrics_name, [silh, ch, db])}
            metrics = dict(zip(metrics_name, [silh, ch, db]))  # per pylint
            mlflow.log_metrics(metrics)
            # metric to minimize
            loss = -silh
            mlflow.sklearn.log_model(
                cluster_pipe,
                artifact_path="model",
            )

        return {"loss": loss, "status": STATUS_OK}

    search_space = hp.choice(
        "clf_type",
        [
            {
                "type": "kmeans",
                # scope.int() solves the issue where hp.quniform returned float, not int
                "n_clusters": scope.int(hp.quniform("knn_n_clusters", 3, 8, 1)),
            },
            {
                "type": "hierarchical",
                "n_clusters": scope.int(hp.quniform("h_n_clusters", 3, 8, 1)),
                "linkage": hp.choice("linkage", ["ward", "average"]),
            },
            {
                "type": "dbscan",
                "eps": hp.lognormal("eps", 0.1, 5.0),
                "min_samples": scope.int(hp.quniform("min_samples", 3, 10, 1)),
                "metric": hp.choice("metric", ["euclidean", "cosine", "manhattan"]),
            },
        ],
    )
    rstate = np.random.default_rng(42)  # for reproducible results
    trials = Trials()
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=trials,
        rstate=rstate,
        return_argmin=True,
    )
    return trials


def _run(data_path, max_evals):
    df = load_data(data_path / "nba_stats.pkl")
    model_search(data=df, num_trials=max_evals)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default="../../data/",
        help="the location where the processed nba career data was saved.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=10,
        help="the number of parameter evaluations for the optimizer to explore.",
    )
    args = parser.parse_args()

    _run(args.data_path, args.max_evals)
