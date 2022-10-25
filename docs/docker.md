# Dockerizing the App

Now that we've got stuff running locally, time to containerize it

Components to dockerize:

- `fetch`
  - calls NBA API for leaguedash
  - transforms and cleans data for modelling
- `train`
  - Trains multiple clusterers and logs results to MLflow
- mlflow?
  - Logs results and artifacts from model training
- streamlit
  - Exposes the downloaded data and clusterer labels to end-user

## Steps for dockerizing to production

1. Run locally in docker
1. Basic security
1. Automate builds
1. Easier debugging and operational correctness
1. Make it reproducible
1. Improve performance by speeding build time and reducing image size

## Execution

Each *service* will have their own `dockerfile`, and stitched together with `docker-compose.yaml`.

More fun with project layouts, I expect.

### Network

MLflow and streamlit will need to share a network. Streamlit will need to expose a public port.

### Volumes

All services need to share a volume

- `fetch` to store API call results and transformed results
- `train` to collect cleaned data
- `mlflow` to store `mlflow.db` backend and `./mlruns` artifacts
- `streamlit` to retrieve the production stage model from `.mlruns` after perusing `mlflow.db`

### Testing

How to test each service? Run a dockerfile for each, and unit-test them.