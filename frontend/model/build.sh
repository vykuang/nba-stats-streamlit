#! /usr/bin/env sh
docker build \
    -t nba-streamlit/model \
    -f model.Dockerfile \
    .
