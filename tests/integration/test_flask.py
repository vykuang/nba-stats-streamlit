"""
Integration test with flask
pytest fixtures runs the local flask backend for these tests to run against
"""

import json

import pytest
from flask import request


@pytest.mark.parametrize(
    ("flask_service", "route", "query_string"),
    (
        (
            "fetch",
            "/fetch",
            {
                "season": "1000-1001",
                "loglevel": "debug",
                "dryrun": "1",
            },
        ),
        (
            "model",
            "/transform",
            {
                "season": "2015-16",
                "loglevel": "debug",
                "dryrun": "1",
            },
        ),
    ),
)
def test_flask_request(client, tmp_path, flask_service, route, query_string):
    """
    GIVEN the flask apps are running
    WHEN a client makes a request to the flask apps
    THEN the apps should receive the intended arguments
    """
    query_string.update({"data_path": tmp_path})
    # GIVEN
    with client(flask_service) as c:
        # WHEN
        response = c.get(route, query_string=query_string)
        assert response.status_code == 200
        # THEN
        # request needs app context
        for item in query_string:
            assert request.args[item] == str(query_string[item])

    request_echo = json.loads(response.data)
    assert request_echo == query_string


def test_flask_transform(client):
    """
    GIVEN app_model is up, and leaguedash.pkl are in data_path
    WHEN client GETs request to "transform"
    THEN app_model transforms the .pkl to modelling-grade input
    """
    # with client("model") as c:
    #     response = c.get()
