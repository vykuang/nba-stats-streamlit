#! /usr/bin/env sh
docker build \
    -t nba-streamlit/model:$1 \
    -f model.Dockerfile \
    .
