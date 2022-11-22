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
ENTRYPOINT [ "poetry", "run", "python", "-m", "flask", "run" ]
# default params for our executable
# docker run <args> will override the CMD params
CMD [ "--host", "0.0.0.0", "--port", "8080" ]
