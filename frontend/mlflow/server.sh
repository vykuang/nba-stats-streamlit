#! /usr/bin/env sh
docker run -d \
    --network=nba-streamlit-mlflow \
    -p 5000:5000 \
    --name mlflow \
    -v nba-mlflow:/mlflow \
    docker-mlflow \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
