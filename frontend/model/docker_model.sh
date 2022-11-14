 #! /usr/bin/env sh
 docker run \
    --network=nba-streamlit-mlflow \
    -v nba-pkl:/data \
    -e "MLFLOW_TRACKING_URI=http://172.18.0.2:5000" \
    --name nba-train \
    nba-streamlit/model train.py --data_path /data --max_evals 3
