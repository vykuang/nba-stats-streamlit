"""
Streamlit demo for NBA stats
Find similar players across different seasons
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.title("NBA Player Comparisons across Eras")


@st.cache
def load_career_stats(season: str, data_path: Path = "../../data/"):
    file_path = data_path / f"leaguedash_merge_{season}.pkl"
    data = pd.read_pickle(file_path)
    return data


data_load_state = st.text("Loading pickle...")

current = load_career_stats(season="2018-19")
comparison = load_career_stats(season="2004-05")

data_load_state = st.text("Finished loading")

if st.checkbox("Show current season stats"):
    st.subheader("Current data")
    st.dataframe(current)

if st.checkbox("Show comparison season stats"):
    st.subheader("Comparison data")
    st.dataframe(comparison)


st.subheader("EDA")


st.subheader("AST AND REB VS PTS DISTRIBUTION")


def make_ast_reb_scatter(stats):
    """Wrapper to make the EDA scatter plot for different seasons"""
    alt_chart = (
        alt.Chart(stats)
        .mark_circle()
        .encode(x="AST", y="REB", size="PTS", color="FG_PCT", tooltip=["FG_PCT", "TOV"])
    )
    return alt_chart


current_scatter = make_ast_reb_scatter(current)
comp_scatter = make_ast_reb_scatter(comparison)
st.altair_chart(current_scatter)
st.altair_chart(comp_scatter)

st.subheader("STAT DISTRIBUTION")
# compares the distribution of each stat between the two seasons
merge_stats = [stat for stat in current.columns if "_merge" in stat]


def make_violin(stats):
    """Wrapper to return altair violinplot via chart().transform_density()"""
    np.sum(stats)
