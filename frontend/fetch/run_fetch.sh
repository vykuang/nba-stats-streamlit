#! /usr/bin/env bash
docker run \
    --network=nba-streamlit-backend \
    -v nba-pkl:/data \
    --mount type=bind,src="$(pwd)",dst=/app \
    --name nba-fetch \
    nba-streamlit/fetch
