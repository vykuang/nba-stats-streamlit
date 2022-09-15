import pytest
from nba_api.stats.endpoints import _base, playercareerstats

from frontend import fetch


class MockCareerResponse:
    @staticmethod
    def get_request():
        return {}


def test_get_career_stats(monkeypatch):
    def mock_get_career(*args, **kwargs):
        return MockCareerResponse()

    # patching the .get_request() class method
    monkeypatch.setattr(
        playercareerstats.PlayerCareerStats,
        "get_request",
        mock_get_career,
    )
    test_id = 2544
    response = fetch.get_career_stats(player_id=test_id)
