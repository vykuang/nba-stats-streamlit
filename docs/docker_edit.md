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

#### mlflow

Making my own mlflow image using 1.28.

Dockerfile does the following:

- 3.9-slim as base
- create `mlflow-venv` to pip install our mlflow
- `pip install mlflow`
- expose port
- use `ENV` vars to set backend-store and artifact-root
- `mlflow server -- ...`

Execution notes:

- Without our venv, we're running pip as root inside our container, which "may result in broken permissions and conflicting behaviours..."

  - see [docs here](https://pip.pypa.io/warnings/venv)

- Activating our newly created venv is a little tricky. Usually we do this:

  ```bash
  python -m venv my-venv-dir
  source my-venv-dir/bin/activate
  ```

  which translates to this in dockerfile:

  ```dockerfile
  # from https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
  FROM python:3.9-slim-bullseye
  RUN python3 -m venv /opt/venv

  # This is wrong!
  # RUN . is how we "source" files in docker
  RUN . /opt/venv/bin/activate

  # Install dependencies:
  COPY requirements.txt .
  RUN pip install -r requirements.txt

  # Run the application:
  COPY myapp.py .
  CMD ["python", "myapp.py"]
  ```

- However that will not work as intended:

  1. Every `RUN` in dockerfile is a separate process; it has no effect on future `RUN` calls. `RUN` . /activate is effective a no-op
  1. `CMD` is also unaffected by `RUN` and thus will not run inside our venv as intended

- \[OPTION 1\] Reference the path to venv in each python call

  - `${MLFLOW_VENV}/bin/pip install --upgrade pip`
  - caveat: subprocess will not run in the venv

- \[OPTION 2\] Activate our venv in each `RUN` and `CMD`

  - `RUN . /opt/venv/bin/activate && pip install`

At its core, `activate` does the following things:

1. finds what shell we're currently in
1. adds `deactivate` function to the shell
1. changes the prompt to include venv's name
1. *unsets* `PYTHONHOME` env var
1. *sets* two env vars: `VIRTUAL_ENV` and `PATH`

The first four has no effect on docker, and so if we can emulate the last step, we're golden. Even for the last step, `VIRTUAL_ENV` has no impact, except when some tools use it to detect whether we're already inside a venv.

So if we set `PATH` correctly, we can activate our venv properly:

```dockerfile
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
```

`CMD` and `RUN` now works with our venv

Run locally with

```bash
docker build -f mlflow.Dockerfile -t docker-mlflow .
docker run -d -p 5000:5000 docker-mlflow
```

#### model

Only sent the `pyproject.toml` in to build the image because I didn't want to build a whole venv just for the .lock file. Build takes longer.

```bash
docker build -f model.Dockerfile -t nba-streamlit/model .
```

Attached `nba-pkl` volume, transform works

To get my `train.py` working I need the following

1. Schedule `train` to run after `transform`. Use a bash script and `&&` to run them sequentially?
   - run `train` on its own by specifying `train.py` in my `docker run` command, since it will overwrite the `transform.py` CMD in my dockerfile
   - set args to `train` as usual
1. Connect the mlflow container via a network
   - create the network - `docker network create nba-streamlit-mlflow`
   - attach the two containers to the network
     - `docker network connect nba-streamlit-mlflow <mlflow_container_name>`
     - the container name can be found with `docker ps`
     - start my `model` container and connect to that network
     - `docker run --network=nba-streamlit-mlflow nba-streamlit/model`
1. Edit the `train` script to connect to remote mlflow tracking
   - set `MLFLOW_TRACKING_URI` env var to the mlflow container:5000 address
   - in `docker run`, set flag `-e "MLFLOW_TRACKING_URI=<mlflow_container_name>:5000"`
   - can use `docker network --alias` to set host name for more legible resolution, much like how docker compose does it
   - otherwise, use `docker network inspect na-streamlit-mlflow` to see what IP to set it to

- Running into an environment var issue. When running `train`:

  ```py
  Model registry functionality is unavailable; got unsupported URI '${BACKEND_URI:-sqlite:////mlflow/backend/mlflow.db}' for model registry data storage.
  ```

  That's what was in the mlflow dockerfile. Inspecting the mlflow container reveals that the correct substitution has taken place, but not when `train` connects to it???

  Also the experiment was successfully logged???

  Try using straight `sqlite:////mlflow/mlflow.db` instead of ENV

  In my `server.sh`, replace the CMD args passed in dockerfile with hard coded args for:

  - backend
  - artifact
  - host (VERY IMPORTANT, OTHERWISE DEFAULTS TO LOCALHOST AND UNABLE TO BE CONNECTED TO)
  - port

- `RESOURCE_DOES_NOT_EXIST: Registered Model with name=nba-player-clusterer not found`
  No registered model to be found with `client.get_latest_versions()`. I need to check whether that registered model exists first.

  Use `client.search_registered_models` to see if one exists. If it does, look for the most recent one's `run_id` to compare with the current. If it does not, register our model (as the first)

  - Need single quotes around the search value in our filter string passed to `search()`
  - In debug, when listing the registered model's properties, do not subscript. Use properties. `.register_model()` returns a single `ModelVersion` entity, not a list.

### Network

MLflow and streamlit will need to share a network. Streamlit will need to expose a public port.

- `nba-streamlit-mlflow` for mlflow and train
- streamlit will also be connected to access mlflow's model registry
- Different network (frontend) for streamlit to expose a public port????

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
- compose `entrypoint` and `command` config can override the dockerfile setting

#### MLflow

Create mlflow volume? or use subdir of existing?

Create new mlflow volume for `./mlruns` and `mlflow.db`

If using local db, use sqlite? I think artifact store should be a different volume, so we can specify a local `./mlruns` dir, or a remote cloud storage if we so wished.

### Train

### Testing

How to test each service? Run a dockerfile for each, and unit-test them. Need to brush up on mlops-zoomcamp's video on integrating containers and testing

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