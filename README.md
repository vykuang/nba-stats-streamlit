# nba-stats-streamlit

A basic end-to-end deployment of a simple model using data from NBA stats.

## Brainstorming

* Given a player's season average, classify the type of player.
    * Train on the 2021-2022 season's averages for each player
    * Clustering to reveal the groups, if any
    * User PUTs their request in the form of a collection of seasonal stat averages, and the model should respond with the predicted class of that player, along with some comparisons from the training set 

## Architecture

* sklearn trains our model
* mlflow tracks experiments
* Flask acts as basic backend (could sub gunicorn if things get serious)
* Streamlit as frontend
* poetry to manage dependencies

## Setup

### Poetry

Kind of a pain on my desktop, I felt. Had issues getting poetry to recognize the version I set with pyenv (3.9.12), and not the system version (3.8.10). Especially strange since `poetry install` could find the correct pyenv version, and installed everything with that, but `poetry env info` then shows 3.8.10. Resolved by using `pyenv shell 3.9.12` prior to `poetry install`. The suggested fixes on poetry docs involved `poetry env use 3.9` to explicitly get poetry to recognize it, but no dice.


