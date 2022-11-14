"""
Trains a selection of clusterer models on our compiled dataset,
and logs the results onto MLflow
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXP_NAME = os.getenv("MLFLOW_EXP_NAME", "nba-leaguedash-cluster")
MLFLOW_REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "nba-player-clusterer")

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXP_NAME)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


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

    def objective(params):
        """Used in conjunction with hyperopt.fmin"""
        with mlflow.start_run():
            # log only the hyperparameters passed
            mlflow.log_params(params)
            cls_type = params["type"]
            logger.info(f"Current trial model type: {cls_type}")

            # can't pass 'type' kwarg to our model; from databricks nb
            del params["type"]

            dbscan = False
            if cls_type == "kmeans":
                cls = cluster.KMeans(**params)
            elif cls_type == "hierarchical":
                cls = cluster.AgglomerativeClustering(**params)
            elif cls_type == "dbscan":
                cls = cluster.DBSCAN(**params)
                dbscan = True
            else:
                raise ValueError(
                    "Model type not accepted; use one of ['kmeans', 'hierarchical', 'dbscan']"
                )
            cluster_pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=4),
                cls,
            )

            cluster_pipe.fit(data)
            # log as metric, not param, so that numeric comparator can be used
            n_labels = len(np.unique(cluster_pipe[-1].labels_))
            if dbscan:  # to account for -1 labels for noisy samples
                n_labels -= 1

            # score using metrics that do not require ground truths
            silh, ch, db = score_cluster(cluster_pipe[-1], data)
            metrics_name = [
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
                "n_labels",
            ]
            # metrics = {name: score for name, score in zip(metrics_name, [silh, ch, db])}
            metrics = dict(zip(metrics_name, [silh, ch, db, n_labels]))  # per pylint
            mlflow.log_metrics(metrics)
            # metric to minimize
            loss = -silh
            model_meta = mlflow.sklearn.log_model(
                cluster_pipe,
                artifact_path="model",
            )
            logger.debug(f"model_meta: {model_meta}")

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
                "eps": hp.lognormal("eps", 0.0, 2.0),
                "min_samples": scope.int(hp.quniform("min_samples", 1, 10, 1)),
                "metric": hp.choice("metric", ["euclidean", "cosine", "manhattan"]),
                "n_jobs": -1,
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


def find_best_model() -> str:
    """
    Search for the best model into production

    Returns:
    --------
    run_id
        str
        Corresponds to the run_id of the best performing model
    """
    exp = client.get_experiment_by_name(MLFLOW_EXP_NAME)
    query = "metrics.n_labels >= 3"
    best_runs = client.search_runs(
        experiment_ids=exp.experiment_id,
        filter_string=query,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=3,
        order_by=["metrics.silhouette_score DESC"],
    )
    for run in best_runs:
        logger.debug(run.info.run_id)

    return best_runs[0].info.run_id


def register_model(run_id: str):
    """
    Register the model and promote to production stage
    """
    ## register
    # /model comes from .log_model path
    model_uri = f"runs:/{run_id}/model"
    latest_vers = client.get_latest_versions(
        name=MLFLOW_REGISTERED_MODEL,
    )
    logger.info(latest_vers)
    run_id_prev = latest_vers[-1].run_id

    if run_id != run_id_prev:
        # model_vers contains meta_data of the registered model,
        # e.g. timestamps, source, tags, desc
        # doc:
        # https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.model_registry.ModelVersion
        model_vers = mlflow.register_model(
            model_uri,
            MLFLOW_REGISTERED_MODEL,
        )
        logger.debug(f"Registered model metadata:\n{model_vers[0]}")
        ## promote
        # returns list[ModelVersion]

        client.transition_model_version_stage(
            name=MLFLOW_REGISTERED_MODEL,
            version=latest_vers[-1].version,
            stage="Production",
            archive_existing_versions=True,
        )
        # ModelVersion of the registered model
        return latest_vers[-1]

    else:
        logger.info(
            f"Previous model (run_id:{run_id}) is still the best performing, no new versions made."
        )
        return None


def _run(
    data_path: Path,
    max_evals: int,
    season: str,
    loglevel: str = "INFO",
):
    """Script to run the functions"""
    num_loglevel = getattr(logging, loglevel, None)

    if not isinstance(num_loglevel, int):
        raise ValueError(f"Invalid log level: {loglevel}")

    logger.setLevel(num_loglevel)
    handler.setLevel(num_loglevel)

    logger.addHandler(handler)

    logger.info(f"Mlflow tracking URI: {MLFLOW_TRACKING_URI}")

    file_path = data_path / f"leaguedash_merge_{season}.pkl"
    logger.info(f"Loading data from: {file_path.resolve()}")
    df = pd.read_pickle(file_path.resolve())

    logger.info(f"Logging all models in {max_evals} trials")
    model_search(data=df.select_dtypes("number"), num_trials=max_evals)

    logger.info("Searching for best logged model")
    run_id = find_best_model()

    logger.info("Registering model")
    model_vers = register_model(run_id)
    if model_vers:
        logger.debug(f"Registered model meta info:\n{model_vers[0]}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default="../../data/",
        help="location where the processed nba career data was saved.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=10,
        help="number of parameter evaluations for the optimizer to explore.",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2018-19",
        help="Season to train our clusterer, e.g. 2018-19.",
    )
    parser.add_argument(
        "--loglevel",
        "-l",
        type=str.upper,
        default="info",
        help="Logger level; use DEBUG, INFO, etc.; lowercase also accepted",
    )
    args = parser.parse_args()

    _run(
        args.data_path,
        args.max_evals,
        args.season,
        args.loglevel,
    )
