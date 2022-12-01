"""
Integration test with flask
pytest fixtures runs the local flask backend for these tests to run against
"""

import json
import logging

import requests

# from frontend.fetch import fetch


def test_flask_fetch(client, tmp_path):
    """Tests for arguments being properly passed to flask"""
    # setup; encode in str for later comparison
    season = "1000-1001"
    data_path = str(tmp_path)
    loglevel = "debug"
    dryrun = "1"
    query_string = {
        "data_path": data_path,
        "dryrun": dryrun,
        "loglevel": loglevel,
        "season": season,
    }
    # execute
    with client("fetch") as c:
        response = c.get(
            "/fetch",
            query_string=query_string,
        )
    # fetch_url = f"http://localhost:8080/fetch?dryrun={dryrun}&season={season}&data_path={data_path}&loglevel={loglevel}"
    # response = requests.get(fetch_url, timeout=5)
    request_echo = json.loads(response.data)
    assert request_echo == query_string


def test_flask_transform():
    """
    Test the transformation step in flask context
    """
    # setup
    season = "2018-19"
    data_path = "/data"
    loglevel = "debug"
    fetch_url = f"http://localhost:8081/transform?season={season}&data_path={data_path}&loglevel={loglevel}"
    response = requests.get(fetch_url, timeout=5)
    logging.info(response)
    assert False
