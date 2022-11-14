FROM python:3.9-slim as base

ENV MLFLOW_VERSION=1.28.0
ENV VIRTUAL_ENV=/mlflow-venv

RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN pip install --upgrade pip \
    && pip install mlflow==${MLFLOW_VERSION} --no-cache-dir

EXPOSE 5000

WORKDIR /mlflow
ENV BACKEND_URI sqlite:////mlflow/backend/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts
ENTRYPOINT [ "mlflow", "server" ]
CMD [ "--backend-store-uri", "${BACKEND_URI:-sqlite:////mlflow/backend/mlflow.db}", "--default-artifact-root", "${ARTIFACT_ROOT}", "--host", "0.0.0.0", "--port", "5000" ]
