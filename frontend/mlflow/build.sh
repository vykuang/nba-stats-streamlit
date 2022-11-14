#! /usr/bin/env sh
docker build \
    -t docker-mlflow \
    -f mlflow.Dockerfile \
    .
