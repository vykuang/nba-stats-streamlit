"""Tests the functions within fetch_league module"""
import logging

import requests
from nba_api.stats.endpoints import leaguedashplayerstats

from frontend.fetch import app


def test_league_json(monkeypatch):
    """Tests that a list is being returned"""

    class MockLeagueDash:
        @staticmethod
        def get_normalized_json():
            return '{"LeagueDashPlayerStats": ["a","b","c"]}'

    def mock_league_dash(*args, **kwargs):
        return MockLeagueDash()

    monkeypatch.setattr(
        leaguedashplayerstats, "LeagueDashPlayerStats", mock_league_dash
    )

    res = app.get_leaguedash_json("Regular Season", "2020-21", "Base")

    assert res
    assert isinstance(res, list)
    assert len(res) == 3


def test_flask_arg():
    """Tests for arguments being properly passed to flask
    Requires flask/fetch to be up"""
    # setup
    season = "1000-1001"
    data_path = "test_path"
    loglevel = "critical"
    dryrun = 1
    fetch_url = f"http://localhost:8080/fetch?dryrun={dryrun}&season={season}&data_path={data_path}&loglevel={loglevel}"
    response = requests.get(fetch_url, timeout=5)
    logging.info(response)
    assert False
