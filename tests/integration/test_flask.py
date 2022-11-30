"""
Integration test with flask
pytest fixtures runs the local flask backend for these tests to run against
"""

import logging

import requests


def test_flask_fetch():
    """Tests for arguments being properly passed to flask"""
    # setup
    season = "1000-1001"
    data_path = "test_path"
    loglevel = "critical"
    dryrun = 1
    fetch_url = f"http://localhost:8080/fetch?dryrun={dryrun}&season={season}&data_path={data_path}&loglevel={loglevel}"
    response = requests.get(fetch_url, timeout=5)
    logging.info(response)
    assert False


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
