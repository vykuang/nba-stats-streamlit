import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, request
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

from . import leaguedash_columns

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXP_NAME = os.getenv("MLFLOW_EXP_NAME", "nba-leaguedash-cluster")
MLFLOW_REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "nba-player-clusterer")
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "sk_model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXP_NAME)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

app_model = Flask(__name__)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Removes extraneous columns from leaguedash,
    and engineers some new features
    """
    result = df.copy()
    result["FG2M"] = result["FGM"] - result["FG3M"]
    result["FG2A"] = result["FGA"] - result["FG3A"]

    result = result.drop(leaguedash_columns.DROP_COLS, axis=1)
    return result


def _reg_post_merge(
    player: pd.Series,
    post_df: pd.DataFrame,
    post_wt: float = 2.0,
) -> pd.Series:
    """Folds regular and post season stats into one via a weight coefficient
    Scoped within transform_leaguedash to make use of post_df"""
    # if either regular or post stats for a given player is missing, use
    # what's present
    # only fold if both are present
    player = player.copy()  # avoids mutating the df as it's being iterated

    post_ids = set(post_df.index)
    if player.name in post_ids:
        post_season = post_df.loc[player.name]
        # initiate merge, since player is present in both reg and post

    else:
        post_season = player

    gp_tot = player["GP"] + post_wt * post_season["GP"]
    for stat in player.index:
        if stat not in leaguedash_columns.PLAYER_BIO:
            player[stat + "_merge"] = (
                player["GP"] / gp_tot * player[stat]
                + post_wt * post_season["GP"] / gp_tot * post_season[stat]
            )
    return player


def reg_post_merge(reg_df, post_df, post_wt: float = 2.0) -> pd.DataFrame:
    """Wrapper to for df.apply(_reg_post_merge)
    This pattern can include the subsequent dropping of columns, for
    better unit testing
    """
    return reg_df.apply(_reg_post_merge, post_df=post_df, post_wt=post_wt, axis=1).drop(
        leaguedash_columns.MERGE_STATS, axis=1
    )


def _leaguedash_rerank(stat: pd.Series) -> pd.Series:
    """Ranks all the values in the given stat column.
    Largest values will be given top ranks
    To be used in df.apply()

    Parameters
    ---------

    stat: pd.Series
        A statistical field with numeric values to be ranked

    Returns
    --------

    stat_rank: pd.Series
        Ranking of the stat Series
    """

    # sort the values
    sorted_stat_index = stat.sort_values(ascending=False).index

    # attach a sequential index to the now sorted values
    sorted_rank = [(rank + 1) for rank in range(len(stat.index))]

    # can't for the life of me figure out how to return my desired column names
    # rename after returning.
    rank_series = pd.Series(
        data=sorted_rank,
        index=sorted_stat_index,
        name=f"{stat.name}_RANK",
    ).reindex(index=stat.index)

    # standardize by dividing by num of players
    rank_series /= len(stat)
    return rank_series


def leaguedash_rerank(merge_df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper for df.apply(_leaguedash_rerank)
    Includes the setup of dropping player_bio columns and subsequent
    replacement/formatting of column names, and dataframe merge
    """
    rerank = merge_df.drop(leaguedash_columns.PLAYER_BIO, axis=1).apply(
        _leaguedash_rerank, axis="index"
    )
    rerank.columns = [col.replace("merge", "RANK") for col in rerank.columns]
    merge_rerank = pd.concat([merge_df, rerank], axis="columns")
    return merge_rerank


def _player_meets_standard(
    player: pd.Series, min_thd: int = 800, gp_thd: int = 40
) -> bool:
    """Does this player pass the minutes or games played threshold?
    Considers the folded minutes/games played
    """
    logger.debug(f"min merge: {player['MIN_merge']}, min_thd: {min_thd}")
    return player["MIN_merge"] >= min_thd or player["GP_merge"] >= gp_thd


def player_meets_standard(
    df: pd.DataFrame, min_thd: int = 800, gp_thd: int = 40
) -> bool:
    """
    Wrapper for df.apply(_player_meets_standard, ...)
    """
    df["gametime_threshold"] = df.apply(
        _player_meets_standard, min_thd=min_thd, gp_thd=gp_thd, axis=1
    )
    return df


def transform_leaguedash(
    reg_df: pd.DataFrame, post_df: pd.DataFrame, post_wt: float = 2.0
) -> pd.DataFrame:
    """Prepares API results for clustering
    Thinking about breaking out the nested functions into top level funcs
    Easier unit testing
    Add input arg of post_ids to each nested func"""
    # feature engineer
    logger.debug("feature engineering...")
    reg_df = feature_engineer(reg_df)
    post_df = feature_engineer(post_df)
    logger.info("feature engineering complete")

    # merge
    logger.debug("Merging regular and post season stats...")
    # drop reg season stats after merging with post season
    # merge_df = reg_df.apply(
    #     reg_post_merge, post_df=post_df, post_wt=post_wt, axis=1
    # ).drop(leaguedash_columns.MERGE_STATS, axis=1)
    merge_df = reg_post_merge(reg_df=reg_df, post_df=post_df, post_wt=post_wt)
    logger.info(f"Merging complete with post_wt = {post_wt:.3f}")
    logger.debug(f"Players post merge: {len(merge_df)}")

    # re-rank using merged stats
    logger.debug("Re-ranking merged stats...")
    # # only rank merged columns, so drop bio before merging
    # merge_ranks = merge_df.drop(leaguedash_columns.PLAYER_BIO, axis=1).apply(
    #     leaguedash_rerank, axis="index"
    # )
    # merge_ranks.columns = [col.replace("merge", "RANK") for col in merge_ranks.columns]
    # logger.info("Re-rank complete")

    # logger.debug(
    #     f"merge_df shape: {merge_df.shape}\nmerge_rank shape: {merge_ranks.shape}"
    # )
    # merge_df = pd.concat([merge_df, merge_ranks], axis="columns")
    merge_df = leaguedash_rerank(merge_df)
    logger.info("Ranks and stats merge complete")
    logger.debug(f"Column count: {len(merge_df.columns)}")

    # filter for minutes and games played
    merge_df = player_meets_standard(merge_df, min_thd=800, gp_thd=40)
    logger.info(f"Number of eligible players: {merge_df['gametime_threshold'].sum()}")
    return merge_df


def dump_pickle(obj, fp: Path) -> None:
    """pickle dump"""
    with open(fp, "wb") as f_out:
        pickle.dump(obj, f_out)


def load_pickle(fp: Path) -> Any:
    """Loads the json pickle and returns as df"""
    with open(fp, "rb") as f_in:
        res = pickle.load(f_in)

    if isinstance(res, list) and isinstance(res[0], dict):
        df = pd.DataFrame.from_dict(res).set_index("PLAYER_ID")
        return df
    else:
        raise TypeError("Expected list of dicts")


def transform(
    season: str, data_path: Path, loglevel: str = "info", overwrite: bool = False
):
    """Loads the pickle for transformation, and stores the result"""
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logger.setLevel(numeric_level)
    handler.setLevel(numeric_level)
    logger.addHandler(handler)

    reg_pkl = data_path / f"leaguedash_regular_{season}.pkl"
    post_pkl = data_path / f"leaguedash_playoffs_{season}.pkl"
    merge_pkl = data_path / f"leaguedash_merge_{season}.pkl"
    if merge_pkl.exists() and not overwrite:
        logger.info(
            f"""
            Merged pickle already exists at:
            {merge_pkl.resolve()}
            Exiting
            """
        )

    else:
        logger.info(f"Loading from {data_path.resolve()}")
        reg_df = load_pickle(reg_pkl)
        post_df = load_pickle(post_pkl)

        logger.debug("Pickles loaded")
        logger.debug(f"Loaded {len(reg_df)} records from reg_pkl")
        logger.debug(f"Loaded {len(post_df)} records from post_pkl")

        merge_df = transform_leaguedash(reg_df, post_df)

        dump_pickle(merge_df, merge_pkl)
        logger.info(f"Results saved to {merge_pkl.resolve()}")
    return str(merge_pkl)


@app_model.route("/transform")
def run_transform():
    """ "Wrapper for transform.py"""
    season = request.args.get("season", default="2015-16", type=str)
    data_path = Path(request.args.get("data_path", default="flask_data", type=str))
    loglevel = request.args.get("loglevel", default="debug", type=str)
    overwrite = request.args.get("overwrite", default=0, type=bool)
    dryrun = request.args.get("dryrun", default=0, type=bool)
    debug_msg = f"""
    season:\t\t{season}
    data_path:\t\t{data_path}
    loglevel:\t\t{loglevel}
    """
    logger.info(debug_msg)
    print(debug_msg)
    if not dryrun:
        result = transform(
            season=season, data_path=data_path, loglevel=loglevel, overwrite=overwrite
        )
    else:
        result = request.args
    return result


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
                artifact_path=MLFLOW_ARTIFACT_PATH,
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


def register_model(run_id: str) -> mlflow.entities.model_registry.ModelVersion:
    """
    Register the model and promote to production stage
    """
    ## register
    # /model comes from .log_model path
    # does the model exist?
    run_id_prev = None
    # note the single quote around the search value
    models = client.search_model_versions(f"name='{MLFLOW_REGISTERED_MODEL}'")
    if models:
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
        model_uri = f"runs:/{run_id}/{MLFLOW_ARTIFACT_PATH}"
        model_vers = mlflow.register_model(
            model_uri,
            MLFLOW_REGISTERED_MODEL,
        )
        # model_vers is NOT SUBSCRIPTABLE. Use the attributes given, e.g. run_id
        logger.info(
            f"""
            Registered model run_id:\t{model_vers.run_id}
            source:\t {model_vers.source}
            """
        )

        ## promote
        # returns list[ModelVersion]
        if models:
            client.transition_model_version_stage(
                name=MLFLOW_REGISTERED_MODEL,
                version=model_vers.version,
                stage="Production",
                archive_existing_versions=True,
            )
        # ModelVersion of the registered model
        return model_vers
    else:
        logger.info(
            f"Previous model (run_id:{run_id}) is still the best performing, no new versions made."
        )
        return None


def train(
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
        logger.info(f"Registered model meta info:\n{model_vers}")


@app_model.route("/model")
def run_model():
    """Wrapper for train.py"""
    season = request.args.get("season", default="2015-16", type=str)
    data_path = Path(request.args.get("data_path", default="flask_data", type=str))
    loglevel = request.args.get("loglevel", default="debug", type=str)
    max_evals = request.args.get("max_evals", default=10, type=int)
    debug_msg = f"""
    season:\t\t{season}
    data_path:\t\t{data_path}
    max_evals:\t\t{max_evals}
    loglevel:\t\t{loglevel}
    """
    logger.info(debug_msg)
    return train(data_path=data_path, max_evals=max_evals, season=season)
