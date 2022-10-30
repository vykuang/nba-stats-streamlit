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

### Poetry

Using Poetry in docker is its own beast.

Sample Dockerfile:

```Docker
FROM python:3.9-slim

WORKDIR /app
ENV POETRY_VERSION=1.2.1

RUN pip install --upgrade pip \
    && apt-get update \
    # && apt install -y curl netcat \
    && curl -sSL https://install.python-poetry.org | \
    python3 - --version ${POETRY_VERSION}

# update PATH
ENV PATH="${PATH}:/root/.poetry/bin"
```

- `curl -sSL` is the official installation method from poetry
  - This separates poetry from the dependencies it manages
- `pip install` is also available: [docs here](https://python-poetry.org/docs/#ci-recommendations)
  - more manual method
  - allows better debugging
  - needs fewer external tools, e.g. `curl`
- pin the version so that upgrades are on our call

### Multi-stage builds

Some disable `venv` in docker as the container is its own isolated environment, but having a venv still serves a purpose because it can *leverage multi-stage builds to reduce image size*.

```
* Build stage installs into the venv, and final stage simply copies the venv over to a smaller image
```
