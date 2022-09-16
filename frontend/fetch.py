import json

from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players


def get_ids(name: str) -> list:
    name_hits = players.find_players_by_full_name(name)
    return name_hits


def get_career_stats(player_id: str, get_request: bool = True) -> dict:
    career_stats = playercareerstats.PlayerCareerStats(
        player_id=player_id,
        per_mode36="Per36",
        get_request=get_request,
    )
    return career_stats.get_normalized_json()


def get_jsons(career_stats):
    career_stats_json = json.loads(career_stats.get_normalized_json())
    reg_season = career_stats_json["SeasonTotalsRegularSeason"][-1]
    post_season = career_stats_json["SeasonTotalsPostSeason"][-1]

    return reg_season, post_season
