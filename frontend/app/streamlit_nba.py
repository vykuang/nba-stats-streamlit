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
import requests
from mlflow.tracking import MlflowClient

import streamlit as st

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
    os._exit(e.code)  # pylint: disable=protected-access

num_loglevel = getattr(logging, args.loglevel, None)

if not isinstance(num_loglevel, int):
    raise ValueError(f"Invalid log level: {args.loglevel}")
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(num_loglevel)
handler.setLevel(num_loglevel)
logger.addHandler(handler)

CHART_WIDTH = 300

# THIS IS A MAGIC CONSTANT AND MUST BE RESOLVED AT DEPLOYMENT
MLFLOW_PATH = os.getenv("MLFLOW_PATH", "../model")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_PATH}/mlflow.db"
)
MLFLOW_EXP_NAME = os.getenv("MLFLOW_EXP_NAME", "nba-leaguedash-cluster")
MLFLOW_REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "nba-player-clusterer")


alt.data_transformers.enable("json")

st.title("NBA Player Comparisons across Eras")

# ----------------------------------------------------------------------------
# USER INPUT
# ----------------------------------------------------------------------------

# season_a_input = "2018-19"
# season_b_input = "2004-05"
# player_name_input = "Fred VanVleet"
season_choices = [f"{year}-{str(year+1)[-2:]}" for year in range(1990, 2022)]
season_a_input = st.selectbox("Pick the season of your choice", season_choices)
player_name_input = st.text_input(label="Input an NBA player's name  from that season")
season_b_input = st.selectbox(
    "Pick the season to compare the selected player against", season_choices
)

# ----------------------------------------------------------------------------
# Loading data based on input
# ----------------------------------------------------------------------------


def fetch_stats(season: str, data_path: str) -> int:
    """Requests the absent season stats from the "fetch" microservice container

    Returns
    --------
    status_code: int
        200 for nominal response
        500 for exception
    """
    # magic variable; need to parametrize the host, port, and route???
    fetch_url = "fetch:8080/fetch"
    query_string = {
        "season": season,
        "data_path": data_path,
    }
    response = requests.get(fetch_url, params=query_string)
    return response.status_code


def transform_stats(season: str, data_path: str) -> int:
    """Requests transformation from "model" microservice container via the
    "/transform" route

    BASICALLY THE SAME FUNC AS FETCH_STATS - REFACTOR!!!

    Returns
    ---------
    status_code: int
    """
    # magic variable
    transform_url = "model:8081/transform"
    query_string = {
        "season": season,
        "data_path": data_path,
    }
    response = requests.get(transform_url, params=query_string)
    return response.status_code


# @st.cache
def load_career_stats(season: str, data_path: str = "../../data/"):
    file_path = Path(data_path) / f"leaguedash_merge_{season}.pkl"
    data = pd.read_pickle(file_path)
    return data


player_season = load_career_stats(season=season_a_input)
comparison = load_career_stats(season=season_b_input)

# combine the two seasons and add 'season' as new feature
# consider appending it in preprocess?
player_season["season"] = player_season.apply(lambda x: season_a_input, axis=1)
comparison["season"] = comparison.apply(lambda x: season_b_input, axis=1)

logger.debug("Loading data")
data_load_state = st.text("Loading pickle...")
logger.info("Loaded data")


def retrieve(model_name: str) -> mlflow.pyfunc.PyFuncModel:
    """
    Retrieves and returns the latest version of the registered model
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # needs additional quotation marks around the filter value
    model_filter = f"name='{model_name}'"
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
            # model_uri=Path(MLFLOW_PATH) / model_uris[-1]
            model_uri=model_uris[-1]
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
            while len(label_name) > 55 or not label_name:
                label_name = "-".join(label_samples.sample(3).values)
            labels[label] = label_name

    data["label_names"] = data["label_pred"].map(labels)
    return labels, data.drop(["rank_agg"], axis=1)


logger.info("Retrieving trained clusterer")
clusterer = retrieve(MLFLOW_REGISTERED_MODEL)

logger.info("Predicting player label for season A")
cluster_labels, player_season = reveal_group(player_season, clusterer)
logger.info("Predicting player label for season B")
comparison = reveal_group(comparison, clusterer, labels=cluster_labels)[1]

data_load_state = st.text("Finished loading")

if st.checkbox("Show current season stats"):
    st.subheader("player_season data")
    st.dataframe(player_season)

if st.checkbox("Show comparison season stats"):
    st.subheader("Comparison data")
    st.dataframe(comparison)


st.subheader("EDA")


st.subheader("AST AND BLK VS PTS DISTRIBUTION")


# @st.cache
def make_ast_reb_scatter(stats, gametime_threshold: bool = True):
    """Wrapper to make the EDA scatter plot for different seasons"""
    if gametime_threshold:
        stats = stats.loc[stats["gametime_threshold"]]
    brush = alt.selection(type="interval")

    base = (
        alt.Chart(stats)
        .mark_point()
        .encode(
            x="AST_merge",
            y="BLK_merge",
            size="PLUS_MINUS_merge",
            color=alt.condition(brush, "PTS_merge:Q", alt.value("grey")),
            tooltip=["PLAYER_NAME"],
        )
        .add_selection(brush)
    )

    ranked_text = (
        alt.Chart(stats)
        .mark_text()
        .encode(y=alt.Y("row_number:O", axis=None))
        .transform_window(row_number="row_number()")
        .transform_filter(brush)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < 20)
    )

    # encoding our data table onto the base
    player_name = ranked_text.encode(text="PLAYER_NAME:N").properties(title="Name")
    team = ranked_text.encode(text="TEAM_ABBREVIATION:N").properties(title="Team")
    pts = ranked_text.encode(text="PTS_merge:Q").properties(title="Points")
    text = alt.hconcat(player_name, team, pts)

    # build chart
    alt_chart = alt.hconcat(
        base,
        text,
    ).resolve_legend(color="independent")
    return alt_chart


logger.info("Preparing scatterplot for season A")
player_season_scatter = make_ast_reb_scatter(player_season)

logger.info("Preparing scatterplot for season B")
comp_scatter = make_ast_reb_scatter(comparison)

logger.debug("Rendering charts in streamlit")
st.altair_chart(player_season_scatter)
st.altair_chart(comp_scatter)

st.subheader("STAT DISTRIBUTION")
# compares the distribution of each stat between the two seasons


# @st.cache
def make_violin(df: pd.DataFrame, var: str, gametime_threshold: bool = True):
    """Wrapper to return altair violinplot via chart().transform_density()

    Parameters:
    -----------

    df: dataframe, wide-format
        Record of all players from the two seasons, must include
        these features:
        - gametime_threshold
        - label_names
        - <stat>_merge variables

    var: str
        column name for which the values will be density transformed

    Returns:
    ---------

    violin: alt.Chart() object
    """
    if gametime_threshold:
        df = df.loc[df["gametime_threshold"]]
    violin = (
        alt.Chart(df, title=var.replace("_merge", ""))
        .transform_density(
            var,
            as_=[var, "density"],
            extent=[df[var].min(), df[var].max()],
            groupby=["season"],
        )
        .mark_area(orient="horizontal")
        .encode(
            y=alt.Y(
                f"{var}:Q",
                axis=alt.Axis(title=None),
            ),
            color="season:N",
            x=alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=False),
            ),
            column=alt.Column(
                "season:N",
                header=alt.Header(
                    title=None,
                    # titleOrient="bottom",
                    # labelAnchor="end",
                    # labelOrient="bottom",
                    # labelAngle=-30,
                    # labelPadding=0,
                ),
            ),
        )
        .properties(width=CHART_WIDTH / 2)
    )

    return violin


def make_chart_arrays(charts: dict, title: str, rowlen: int = 3):
    """Given a dict of altair charts, display in an array"""
    base = alt.vconcat(title=title)
    while charts:
        rows = alt.hconcat()
        for _ in range(rowlen):
            if charts:
                rows |= charts.popitem()[1]
        base &= rows

    return base


merge_stats = [stat for stat in player_season.columns if "_merge" in stat]
src = pd.concat([player_season, comparison], axis=0)

logger.info("Creating violin charts")
violins = {stat: make_violin(df=src, var=stat) for stat in merge_stats}

violin_title = "Traditional Stat Distribution between Seasons"
violins = make_chart_arrays(violins, violin_title)

violins.configure_legend(
    strokeColor="gray",
    fillColor="#EEEEEE",
    padding=10,
    cornerRadius=5,
    orient="top-right",
)
# player_season_pts = make_violin(player_season, "PTS_merge")
logger.debug("Rendering violin charts")
st.altair_chart(violins)

# ----------------------------------------------------------------------------
# User chooses the player for season comparison here
# App uses the pre-trained model to return three similar players
# and visualizes the comparison in a series of bar graphs
# ----------------------------------------------------------------------------
st.subheader(f"{player_name_input} Player Comparisons")

# create comparison column to sort "similarity"
# across season, we look not to indiv. stats, but to overall
# impact and playtime
src["comparison_rank"] = src["PLUS_MINUS_RANK"] + src["MIN_RANK"]
src["selected_player"] = src["PLAYER_NAME"].apply(lambda x: x == player_name_input)
player_stat = src.loc[src["PLAYER_NAME"] == player_name_input]
# returns a pd.Series of len 1
player_label = player_stat["label_pred"].values[0]

comparison_pool = src[
    (src["season"] == season_b_input)
    & (src["label_pred"] == player_label)
    & (src["gametime_threshold"])
]


similarity_index = (
    (comparison_pool["comparison_rank"] - player_stat["comparison_rank"].values)
    .abs()
    .sort_values(ascending=True)
    .index
)
similars = comparison_pool.loc[similarity_index].head(2)


def make_stat_bar(
    bar_stat: str,
    player: pd.DataFrame,
    analogues: pd.DataFrame,
    season_b: pd.DataFrame,
):
    """Creates a bar graph for five players given the stat:
    1. Selected player
    2. 2 x Similar players
    3. 1 x Stat-Top
    4. 1 x Stat-Bottom

    Parameters:
    -----------
    bar_stat: str
        "<stat>_merge"

    player, analogues: pd.DataFrame
        Record of the selected player and similar player stats

    season_b: pd.DataFrame
        Record of all players from season_b sharing same label as selected player
        Used to find the top/bottom player for each given stat

    Returns
    -------
    stat_bar: alt.Chart() object
        bar graph of the five players and the given stat
    """
    bar_ranked = season_b[bar_stat].sort_values(ascending=False).index
    top = season_b.loc[bar_ranked].head(1)
    bot = season_b.loc[bar_ranked].tail(1)
    df_stat = pd.concat([player, analogues, top, bot], axis=0)
    stat_bar = (
        alt.Chart(df_stat, title=bar_stat.replace("_merge", ""))
        .mark_bar(width=30)
        .encode(
            y=alt.Y(
                f"{bar_stat}:Q",
                axis=alt.Axis(
                    title=None,
                ),
            ),
            x=alt.X(
                "PLAYER_NAME:N",
                # sort=df_stat.sort_values(by=bar_stat)['PLAYER_NAME'].values,
                # sort='ascending', # sorts X-axis string vals
                sort="y",
                axis=alt.Axis(
                    labels=True,
                    title=None,
                    labelAngle=-30,
                ),
            ),
            color=alt.Color("selected_player:N", legend=None),
        )
        .properties(width=CHART_WIDTH)
    )
    return stat_bar


logger.info("Creating bar graphs for each stat")
bars = {
    stat: make_stat_bar(stat, player_stat, similars, comparison_pool)
    for stat in merge_stats
}
bar_title = "Stat Comparison between chosen player and selected players from comparison season".title()
bar_graphs = make_chart_arrays(bars, title=bar_title)
logger.info("Rendering in streamlit")
st.altair_chart(bar_graphs)
