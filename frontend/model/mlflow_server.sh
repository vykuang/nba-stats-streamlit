#! /usr/bin/env sh
poetry run \
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
