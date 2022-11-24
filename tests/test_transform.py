"""
Tests functions from transform
"""
import logging

import pytest
import requests

from frontend.model import transform


def test_transform_flask():
    """"""
    # setup
    season = "2018-19"
    data_path = "/data"
    loglevel = "debug"
    fetch_url = f"http://localhost:8081/transform?season={season}&data_path={data_path}&loglevel={loglevel}"
    response = requests.get(fetch_url, timeout=5)
    logging.info(response)
    assert False
