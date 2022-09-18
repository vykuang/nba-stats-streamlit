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


def test_fold_post_stats():
    """Tests merge stat function"""
    # setup
    reg_season = {"PLAYER_ID": 123, "GP": 30, "FGM": 1, "FGA": 10}
    post_season = {"PLAYER_ID": 123, "GP": 10, "FGM": 9, "FGA": 10}
    merge_list = reg_season.keys()

    # test
    merged = fetch.fold_post_stats(reg_season, post_season, merge_list, 3.0)
    assert merged.keys() == merge_list
    assert merged["FGM"] == 5
    assert merged["PLAYER_ID"] == 123


def test_player_standard():
    """Tests the player check function"""
    # setup
    num = 20
    reg = {"MIN": num / 2, "GP": num / 2}
    post = reg.copy()

    assert fetch.player_meets_standard(reg, post, min_thd=num, gp_thd=num)
    assert fetch.player_meets_standard(reg, post, min_thd=num + 1, gp_thd=num)
    assert fetch.player_meets_standard(reg, post, min_thd=num, gp_thd=num + 1)
    assert not fetch.player_meets_standard(reg, post, num + 1, num + 1)


def test_merge_career(monkeypatch):
    """Tests merge_career_stats by monkeypatching the various subfuncs and substituting in
    Test results
    """
    # setup
    def mock_pickle():
        pkl = {
            "SeasonTotalsRegularSeason": [{"PLAYER_ID": 1, "GP": 80, "MIN": 500}],
            "SeasonTotalsPostSeason": [{"PLAYER_ID": 1, "GP": 8, "MIN": 40}],
        }

        return pkl

    monkeypatch.setattr(
        fetch,
        "load_pickle",
        mock_pickle,
    )
