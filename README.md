# nba-stats-streamlit

A basic end-to-end deployment of a simple model using data from NBA stats.

## Brainstorming

- Given a player's season average, classify the type of player.
  - Train on the 2021-2022 season's averages for each player
  - Clustering to reveal the groups, if any
  - User PUTs their request in the form of a collection of seasonal stat averages, and the model should respond with the predicted class of that player, along with some comparisons from the training set

## Architecture

- sklearn trains our model
- mlflow tracks experiments
- Flask acts as basic backend (could sub gunicorn if things get serious)
- Streamlit as frontend
- poetry manages dependencies

## Setup
