import json
from pathlib import Path

from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players

from frontend.fetch import app


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
    response = app.get_career_stats(player_id=test_id)
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
    merged = app.fold_post_stats(reg_season, post_season, merge_list, 3.0)
    assert merged.keys() == merge_list
    assert merged["FGM"] == 5
    assert merged["PLAYER_ID"] == 123


def test_merge_career(tmp_path, monkeypatch):
    """Tests merge_career_stats by monkeypatching the various subfuncs and substituting in
    Test results
    """
    # setup
    id_lebron = 2544

    def mock_pickle(*args, **kwargs):
        pkl = {
            1: {
                "SeasonTotalsRegularSeason": [
                    {"PLAYER_ID": id_lebron, "GP": 80, "MIN": 500}
                ],
                "SeasonTotalsPostSeason": [
                    {"PLAYER_ID": id_lebron, "GP": 8, "MIN": 40}
                ],
            }
        }

        return pkl

    monkeypatch.setattr(
        app,
        "load_pickle",
        mock_pickle,
    )

    def mock_fold(*args, **kwargs) -> dict:
        sample_fold = {"PLAYER_ID": id_lebron, "GP": 88, "MIN": 540}

        return sample_fold

    monkeypatch.setattr(
        app,
        "fold_post_stats",
        mock_fold,
    )
    result = app.merge_career_stats(tmp_path).to_dict(orient="dict")

    assert Path(tmp_path / "nba_stats.pkl").exists()
    assert result == {
        "GP": {id_lebron: 88},
        "MIN": {id_lebron: 540},
        "full_name": {id_lebron: "LeBron James"},
    }


def test_get_json():
    """Tests for 1) is input json.loads()-able and
    2) whether it contains the relevant keys, and
    3) validity of the value of those keys"""
    # setup
    # mock_stats = """{"SeasonTotalsRegularSeason": [False, True], "SeasonTotalsPostSeason": [False, True]}"""
    mock_stats = (
        '{"SeasonTotalsRegularSeason": [1, 2], "SeasonTotalsPostSeason": [2,3]}'
    )
    # execute
    reg, post = app.get_jsons(mock_stats)

    # assert
    assert reg == 2
    assert post == 3


def test_append_json(tmp_path):
    """Tests the incremental json updating function"""
    # setup
    json_path = tmp_path / "test_append.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({}, file, indent=4)

    # note that using numbers anywhere will result in str being read back
    test_dict = {"id": [{"a": ["x"]}, {"b": ["y"]}, {"c": ["z"]}]}
    # diff ID, same content as test_dict
    test_dict_add = {"id2": test_dict["id"]}
    # run func
    app.append_json(json_path, test_dict)

    # assert
    with open(json_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    assert result == test_dict

    app.append_json(json_path, test_dict_add)

    with open(json_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    assert result["id2"] == test_dict["id"]
