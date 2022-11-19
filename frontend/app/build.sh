#! /usr/bin/env bash
docker build \
    -f streamlit.Dockerfile \
    -t nba-streamlit/app:$1 \
    .
