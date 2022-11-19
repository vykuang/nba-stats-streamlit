#! /usr/bin/env bash
docker run \
    --network=nba-streamlit-mlflow  \
    -v nba-pkl:/data \
    -v nba-mlflow:/mlflow \
    --mount type=bind,src="$(pwd)",dst=/app \
    -p 8501:8501 \
    -e "MLFLOW_PATH=/mlflow" \
    -d \
    --name nba-streamlit \
    nba-streamlit/app:$1
