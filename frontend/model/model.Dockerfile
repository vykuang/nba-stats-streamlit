FROM python:3.9-slim as base

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

WORKDIR /model

# install dependencies
# no lock file because I didn't want to create a whole venv for it
COPY pyproject.toml /model/

# [optional] validate project config
RUN poetry check

# install dependencies
RUN poetry install --no-interaction --no-cache --without dev

# copies src code to container
COPY . /model/

# run the app
ENTRYPOINT [ "poetry", "run", "python" ]

CMD [ "transform.py", "--season", "2018-19", "--data_path", "/data" ]
