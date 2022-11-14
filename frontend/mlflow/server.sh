#! /usr/bin/env sh
docker run -d --rm \
    --network=nba-streamlit-mlflow \
    -p 5000:5000 \
    --name mlflow \
    -v nba-mlflow:/mlflow \
    docker-mlflow