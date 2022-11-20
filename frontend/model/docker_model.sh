 #! /usr/bin/env sh
 docker run \
    --network=nba-streamlit-mlflow \
    -v nba-pkl:/data \
    -v mlflow-artifacts:/mlflow/mlruns \
    --mount type=bind,src="$(pwd)",dst=/model \
    -e "MLFLOW_TRACKING_URI=http://172.18.0.2:5000" \
    --name nba-train \
    nba-streamlit/model \
        train.py --data_path /data --season "2018-19" --max_evals 3
        # transform.py --data_path /data  --season "2004-05"
