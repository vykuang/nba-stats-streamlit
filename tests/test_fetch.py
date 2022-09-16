from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players

from frontend import fetch


def test_get_career_stats(monkeypatch):
    def mock_get_career(*args, **kwargs):
        """When PlayCareerStats calls self.get_response(),
        it will instead call this function, bypassing the
        NBA API call.
        The *args, **kwargs param are defined above so that any args
        passed to the func in original context can be accepted
        without fault
        """
        # pass # unnecessary pass
        # return MockCareerResponse()

    def mock_get_json(*args, **kwargs):
        """When we call career_stats.get_normalized_json(), it will
        call this func instead
        """
        return {"mock_header": ["mock1", "mock2"]}

    # patching the .get_request() instance method
    monkeypatch.setattr(
        playercareerstats.PlayerCareerStats,
        "get_response",
        mock_get_career,
    )
    monkeypatch.setattr(
        playercareerstats.PlayerCareerStats,
        "get_normalized_json",
        mock_get_json,
    )
    test_id = 2544
    response = fetch.get_career_stats(player_id=test_id)
    assert response["mock_header"][0] == "mock1"


def test_get_ids(monkeypatch):
    """Test retrieving player info from static endpoint"""

    def mock_get_id(*args, **kwargs) -> dict:
        test_player = {
            "id": 123,
            "first_name": "first",
            "last_name": "last",
            "full_name": "first last",
        }
        return test_player
        # return MockPlayersResponse()

    monkeypatch.setattr(
        players,
        "find_players_by_full_name",
        mock_get_id,
    )

    player = players.find_players_by_full_name()
    assert player["id"] == 123 and player["full_name"] == "first last"
