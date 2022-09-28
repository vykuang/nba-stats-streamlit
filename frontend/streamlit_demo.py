import numpy as np
import pandas as pd

import streamlit as st

st.title("Uber pickups in NYC")

DATE_COLUMN = "date/time"
DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/"
    "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

# toggle the raw data with a button
if st.checkbox("Show raw data"):
    # section header
    st.subheader("Raw data")

    # previewing data before working on it
    # st.write can render almost anything; st.dataframe is tailored for df
    st.write(data)  # defaults to 10 rows, but all rows can be scrolled

# histogram
# header
st.subheader("Number of pickups by hour")
# generate the actual histogram
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]

# use streamlit to visualize
st.bar_chart(hist_values)

# new section - plotting on map
st.subheader("Map of pickups")

# uses the lat/long to plot on an interactive openstreetmap
# st.map(data)

# filter to view only hour 17
# hour_to_filter = 17

# filter as a slider
hour_to_filter = st.slider("hour", 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f"Map of all pickups at {hour_to_filter}:00")
st.map(filtered_data)

# st.pydeck_chart for more complex maps
