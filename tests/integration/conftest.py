import pytest

from frontend.fetch import fetch
from frontend.model import model


@pytest.fixture
def make_app():
    def _make_app(name: str):
        if name == "fetch":
            return fetch.app
        elif name == "model":
            return model.app_model
        else:
            return None

    return _make_app


@pytest.fixture
def client(make_app):
    def _client(name: str):
        return make_app(name).test_client()

    return _client


@pytest.fixture
def runner(make_app):
    def _runner(name: str):
        return make_app(name).test_cli_runner()

    return _runner
