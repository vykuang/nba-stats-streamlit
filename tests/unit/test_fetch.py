"""Tests the functions within fetch_league module"""
import pytest
from nba_api.stats.endpoints import leaguedashplayerstats

from frontend.fetch import fetch


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

    res = fetch.get_leaguedash_json("Regular Season", "2020-21", "Base")

    assert res
    assert isinstance(res, list)
    assert len(res) == 3


@pytest.mark.api
def test_league_api():
    """Tests the NBA API by making an actual request"""
    res = fetch.get_leaguedash_json(
        season_type="Regular Season",
        season="2022-23",
    )
    assert res
    assert isinstance(res, list)
    assert len(res[0]) == 68
