from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

import streamlit as st

st.title("NBA Player Comparisons across Eras")


@st.cache
def load_career_stats(data_path: Path = "../../data/nba_stats.pkl"):
    data = pd.read_pickle(data_path)
    return data


data_load_state = st.text("Loading pickle...")

nba_stats = load_career_stats()

data_load_state = st.text("Finished loading")

if st.checkbox("Show raw stats"):
    st.subheader("Raw data")
    st.dataframe(nba_stats)

st.subheader("EDA")

st.subheader("PTS")
# reb_filter = st.slider("REBs", 0, 15, 5)
# reb_filtered = nba_stats[np.around(nba_stats["OREB"] + nba_stats["DREB"]) == reb_filter]
# reb_filtered_pts = reb_filtered.loc[:, "PTS"]
# hist_points = np.histogram(reb_filtered_pts, bins=10, range=(0, 35))[0]


hist_points = np.histogram(nba_stats["PTS"], bins=range(0, 35))[0]

st.bar_chart(hist_points)

st.subheader("PTS, AST, REB")
alt_chart = (
    alt.Chart(nba_stats)
    .mark_circle()
    .encode(x="AST", y="REB", size="PTS", color="FG_PCT", tooltip=["FG_PCT", "TOV"])
)
st.altair_chart(alt_chart)