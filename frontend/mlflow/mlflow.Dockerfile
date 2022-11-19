FROM python:3.9-slim as base

ENV MLFLOW_VERSION=1.28.0
ENV VIRTUAL_ENV=/mlflow-venv

RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN pip install --upgrade pip \
    && pip install mlflow==${MLFLOW_VERSION} --no-cache-dir

WORKDIR /mlflow
ENV MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/backend/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/mlruns
ENV MLFLOW_HOST=0.0.0.0
ENV MLFLOW_PORT=5000

EXPOSE ${MLFLOW_PORT}

ENTRYPOINT exec mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} \
    --host ${MLFLOW_HOST} \
    --port ${MLFLOW_PORT}
# CMD [ "--help" ]
# use env var to set defaults
# see https://mlflow.org/docs/latest/cli.html#mlflow-server for env vars
# CMD --host ${MLFLOW_HOST:-0.0.0.0} --port ${MLFLOW_PORT:-5000}
