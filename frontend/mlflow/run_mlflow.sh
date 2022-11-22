#! /usr/bin/env sh
docker run -d --rm \
    --network=nba-streamlit-mlflow \
    -p 5000:5000 \
    --name mlflow \
    -v mlflow-db:/mlflow/backend \
    -v mlflow-artifacts:/mlflow/mlruns \
    docker-mlflow
