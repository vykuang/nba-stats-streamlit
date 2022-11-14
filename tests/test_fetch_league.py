"""Tests the functions within fetch_league module"""
from nba_api.stats.endpoints import leaguedashplayerstats

from frontend.fetch import fetch_league


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

    res = fetch_league.get_leaguedash_json("Regular Season", "2020-21", "Base")

    assert res
    assert isinstance(res, list)
    assert len(res) == 3
