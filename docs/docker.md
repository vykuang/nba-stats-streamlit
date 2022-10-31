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

### Images

Images will be created based on the task dependencies

- `fetch` requires `nba_stats` and little else
- `train` and `transform` needs the machine learning suite
  - `transform` requires `pandas`
- `mlflow` may require its own
  - at this point `mlflow` will use local sqlite `mlflow.db` and `./mlruns` to record the runs
  - create new volumes to store and persist the records
  - but if I already have the postgres10 image I may as well use it???
  - Use local to test the docker image
  - Then test against network postgres container
- `streamlit` is standalone

### Network

MLflow and streamlit will need to share a network. Streamlit will need to expose a public port.

### Volumes

All services need to share a volume

- `fetch` to store API call results and transformed results
- `train` to collect cleaned data
- `mlflow` to store `mlflow.db` backend and `./mlruns` artifacts
- `streamlit` to retrieve the production stage model from `.mlruns` after perusing `mlflow.db`

#### Fetch

Let's try using volume as our storage destination. We'll create a volume via docker CLI, and attach it to the `fetch` volume. This way we can mimic the volume attachment in docker-compose.

```bash
docker volume create nba-pkl
docker run -v nba-pkl:/data nba-streamlit/fetch:latest
```

- `/data` is in the root dir of the container, and is also specified in dockerfile's `CMD`
- Dockerfile requires `poetry run` prior to `python` so that it can use the poetry installed venv
- `ENTRYPOINT` and `CMD` both start in the path set by `WORKDIR`, i.e. `/app`

### Testing

How to test each service? Run a dockerfile for each, and unit-test them.

### Poetry

Using Poetry in docker is its own beast.

Sample Dockerfile:

```Docker
# from https://stackoverflow.com/a/72465422/5496416
FROM python:3.9-slim as base

# configure poetry to install with pip
# see https://python-poetry.org/docs/#ci-recommendations
ENV POETRY_VERSION=1.2.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv

# cache and venv location
ENV POETRY_CACHE_DIR=/opt/.cache

# builder stage for poetry installation
FROM base as builder

# new venv for just poetry, and install poetry with pip; see official doc link
RUN python3 -m venv $POETRY_VENV \
    && ${POETRY_VENV}/bin/pip install --upgrade pip \
    && ${POETRY_VENV}/bin/pip install poetry==${POETRY_VERSION}

# final stage for image
FROM base as app

# copies the poetry installation from builder img to app img
COPY --from=builder ${POETRY_VENV} ${POETRY_VENV}

# allows poetry to be recognized in shell
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

# install dependencies
COPY poetry.lock pyproject.toml /app/

# [optional] validate project config
RUN poetry check

# install dependencies
RUN poetry install --no-interaction --no-cache --without dev

# copies src code to container
COPY . /app/

# run the app
CMD [ "poetry", "--version" ]
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

- Build stage installs into the venv, and final stage simply copies the venv over to a smaller image
