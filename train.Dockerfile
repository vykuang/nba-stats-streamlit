FROM ghcr.io/astral-sh/uv:python3.11-trixie

# Change the working directory to the `app` directory
WORKDIR /app
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# bind mount your code
# Set entrypoint (example: Jupyter, Streamlit, or your app)
CMD ["uv", "run", "python", "test_mlflow.py"]