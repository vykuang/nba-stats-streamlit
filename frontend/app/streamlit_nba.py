"""
Streamlit demo for NBA stats
Find similar players across different seasons
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import altair as alt
import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser(
    description="Streamlit app to compare NBA players and seasons"
)
parser.add_argument(
    "--loglevel",
    "-l",
    type=str.upper,
    default="INFO",
    help="Logger level; set to DEBUG, INFO, etc. Accepts lowercase",
)
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)

num_loglevel = getattr(logging, args.loglevel, None)

if not isinstance(num_loglevel, int):
    raise ValueError(f"Invalid log level: {args.loglevel}")
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(num_loglevel)
handler.setLevel(num_loglevel)
logger.addHandler(handler)

REL_PATH = "../model"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{REL_PATH}/mlflow.db"
)
MLFLOW_EXP_NAME = os.getenv("MLFLOW_EXP_NAME", "nba-leaguedash-cluster")
MLFLOW_REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "nba-player-clusterer")


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXP_NAME)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

st.title("NBA Player Comparisons across Eras")


@st.cache
def load_career_stats(season: str, data_path: Path = "../../data/"):
    file_path = Path(data_path) / f"leaguedash_merge_{season}.pkl"
    data = pd.read_pickle(file_path)
    return data


logger.debug("Loading data")
data_load_state = st.text("Loading pickle...")
player_season = load_career_stats(season="2018-19")
comparison = load_career_stats(season="2004-05")
logger.info("Loaded data")


@st.cache
def retrieve() -> mlflow.pyfunc.PyFuncModel:
    """
    Retrieves and returns the latest version of the registered model
    """
    # needs additional quotation marks around the filter value
    model_filter = f"name='{MLFLOW_REGISTERED_MODEL}'"
    mv_search = client.search_model_versions(model_filter)
    logger.info(f"MLflow model versions returned: {len(mv_search)}")

    model_uris = [mv.source for mv in mv_search if mv.current_stage == "Production"]

    if model_uris:
        logger.info(f"selected model URI: {model_uris[-1]}")
        # alternatively model_uri could also be direct path to s3 bucket:
        # s3://{MLFLOW_ARTIFACT_STORE}/<exp_id>/<run_id>/artifacts/models
        # the models:/model_name/Production uri is only useable if MLflow server is up
        retrieve_model = mlflow.pyfunc.load_model(
            # model_uri=f'models:/{MLFLOW_REGISTERED_MODEL}/Production'
            model_uri=Path(REL_PATH)
            / model_uris[-1]
        )
        return retrieve_model
    else:
        logger.critical("No production stage model found")
        raise Exception("No model found in production stage")


@st.cache
def reveal_group(
    data, model: mlflow.pyfunc.PyFuncModel, labels: dict = None
) -> Tuple[dict, pd.DataFrame]:
    """
    Use a logged model to visualize what label's what
    """
    data = data.copy()
    label_preds = model.predict(data.select_dtypes("number"))
    data["label_pred"] = label_preds
    # min to incentivize recognizability
    # plus_minus for substance
    data["rank_agg"] = np.sum(
        [
            # data.PTS_RANK,
            # data.DREB_RANK,
            # data.OREB_RANK,
            # data.AST_RANK,
            data.MIN_RANK,
            data.PLUS_MINUS_RANK,
        ],
        axis=0,
    )
    # print(data.columns)

    df_sort = data.groupby("label_pred").apply(
        lambda x: x.sort_values(["rank_agg"], ascending=True)
    )
    # df_samp = []
    if not labels:
        labels = {}
        for label in np.unique(label_preds):
            label_samples = df_sort.loc[[label], "PLAYER_NAME"].head(10).sample(3)
            label_name = ""
            while len(label_name) > 50 or not label_name:
                label_name = "-".join(label_samples.sample(3).values)
            labels[label] = label_name

    data["label_names"] = data["label_pred"].map(labels)
    return labels, data.drop(["rank_agg"], axis=1)


clusterer = retrieve()
cluster_labels, player_season = reveal_group(player_season, clusterer)
_, comparison = reveal_group(comparison, clusterer, labels=cluster_labels)

data_load_state = st.text("Finished loading")

if st.checkbox("Show current season stats"):
    st.subheader("player_season data")
    st.dataframe(player_season)

if st.checkbox("Show comparison season stats"):
    st.subheader("Comparison data")
    st.dataframe(comparison)


st.subheader("EDA")


st.subheader("AST AND REB VS PTS DISTRIBUTION")


def make_ast_reb_scatter(stats, gametime_threshold: bool = True):
    """Wrapper to make the EDA scatter plot for different seasons"""
    if gametime_threshold:
        stats = stats.loc[stats["gametime_threshold"]]
    alt_chart = (
        alt.Chart(stats)
        .mark_circle()
        .encode(
            x="AST_merge",
            y="DREB_merge",
            size="PTS_merge",
            color="PLUS_MINUS_merge",
            tooltip=["PLAYER_NAME"],
        )
    )
    return alt_chart


player_season_scatter = make_ast_reb_scatter(player_season)
comp_scatter = make_ast_reb_scatter(comparison)
st.altair_chart(player_season_scatter)
st.altair_chart(comp_scatter)

st.subheader("STAT DISTRIBUTION")
# compares the distribution of each stat between the two seasons
merge_stats = [stat for stat in player_season.columns if "_merge" in stat]


def make_violin(stat: pd.DataFrame, name: str, gametime_threshold: bool = True):
    """Wrapper to return altair violinplot via chart().transform_density()
    'name:Q' or 'name:N' is a shorthand to encode the datatypes for altair
    Q - quantitatve, N - nominal
    """
    if gametime_threshold:
        stat = stat.loc[stat["gametime_threshold"]]
    violin = (
        alt.Chart(stat)
        .transform_density(
            name,
            as_=[name, "density"],
            extent=[stat[name].min(), stat[name].max()],
            groupby=["label_names"],
        )
        .mark_area(orient="horizontal")
        .encode(
            y=f"{name}:Q",
            color="label_names:N",
            x=alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=False),
            ),
            column=alt.Column(
                "label_names:N",
                header=alt.Header(
                    titleOrient="bottom",
                    labelAnchor="end",
                    labelOrient="bottom",
                    labelAngle=-30,
                    labelPadding=0,
                ),
            ),
        )
        .properties(width=100)
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )

    return violin


player_season_pts = make_violin(player_season, "PTS_merge")
st.altair_chart(player_season_pts)
