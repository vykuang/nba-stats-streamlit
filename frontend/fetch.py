import argparse
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players


def get_career_stats(player_id: str, get_request: bool = True) -> dict:
    """Calls NBA API for playercareerstats"""
    career_stats = playercareerstats.PlayerCareerStats(
        player_id=player_id,
        per_mode36="Per36",
        get_request=get_request,
    )
    return career_stats.get_normalized_json()


def get_jsons(career_stats: dict) -> Tuple[dict, dict]:
    """Takes in playercareeer dataset as dict and returns
    the regular and post season totals, for the most recent season"""
    reg_season = career_stats["SeasonTotalsRegularSeason"][-1]
    post_season = career_stats["SeasonTotalsPostSeason"][-1]

    return (reg_season, post_season)


def player_meets_standard(
    reg: dict, post: dict, min_thd: int = 500, gp_thd: int = 40
) -> bool:
    """Does this player have >= 500 min or >= 40 games played?"""
    return bool(reg["MIN"] + post["MIN"] >= min_thd or reg["GP"] + post["GP"] >= gp_thd)


def fold_post_stats(
    reg_season: dict, post_season: dict, merge_stats: list, post_wt: float = 2.0
) -> dict:
    """Merge stats proportionally via games played"""
    if player_meets_standard(reg_season, post_season):
        merged = {}

        gp_tot = reg_season["GP"] + post_wt * post_season["GP"]
        for stat in merge_stats:
            merged[stat] = (
                reg_season["GP"] / gp_tot * reg_season[stat]
                + post_wt * post_season["GP"] / gp_tot * post_season[stat]
            )

        merged["PLAYER_ID"] = reg_season["PLAYER_ID"]
        # merged['SEASON_ID'] = reg_season['SEASON_ID']
        return merged


def dump_pickle(obj, fp: Path) -> None:
    """pickle dump"""
    with open(fp, "wb") as f_out:
        pickle.dump(obj, f_out)


def load_pickle(fp: Path) -> Any:
    """pickle load"""
    with open(fp, "rb") as f_in:
        return pickle.load(f_in)


def fetch(pkl_path: Path = Path("../data")) -> None:
    """Calls stats.nba API for each active player and stages
    in local dir
    This is separate from preprocessing since API calls are an
    expensive resource and should not be retried often
    """
    active = players.get_active_players()
    player_dict = {}
    active_stats = {}
    for player in active:
        # build map to find player name
        player_dict[player["id"]] = player["full_name"]
        # calls stats.nba API for each active player
        career_stats = get_career_stats(player["id"])
        active_stats[player["id"]] = career_stats

        # avoids hammering the API and being blocked
        wait = random.gammavariate(alpha=9.0, beta=0.4)
        time.sleep(wait)

    dump_pickle(active_stats, pkl_path / "careerstats.pkl")
    dump_pickle(player_dict, pkl_path / "player_map.pkl")


def merge_career_stats(pkl_path: Path = Path("../data")) -> pd.DataFrame:
    """Reads the staged .pkl and preprocess into pd.DataFrame for our model"""

    logger = logging.getLogger("merge")
    careers = load_pickle(pkl_path / "careerstats.pkl")
    # remove collinear stats
    merge_stats = [
        # 'FGM',
        "FGA",
        "FG_PCT",
        # 'FG3M',
        "FG3A",
        "FG3_PCT",
        # 'FTM',
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
    ]

    reg_post_merge = []
    for player_id in careers:
        reg_season, post_season = get_jsons(career_stats=careers[player_id])
        merged = fold_post_stats(
            reg_season=reg_season,
            post_season=post_season,
            merge_stats=merge_stats,
        )
        if merged:
            reg_post_merge.append(merged)

    logger.debug(f"merged dicts: , {reg_post_merge}")
    df = pd.DataFrame.from_dict(
        reg_post_merge,
        orient="columns",
    ).set_index("PLAYER_ID")
    logger.debug(
        f"index: {list(df.index)}, content: {df.iloc[:2].to_dict(orient='dict')}"
    )
    dump_pickle(df, pkl_path / "nba_stats.pkl")
    return df


def prepare(pkl_path: Path) -> pd.DataFrame:
    if not pkl_path.exists():
        pkl_path.mkdir(parents=True, exist_ok=False)

    fetch(pkl_path=pkl_path)
    return merge_career_stats(pkl_path=pkl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="NBA career stats fetch")
    parser.add_argument(
        "--pkl_path",
        "-p",
        type=Path,
        default="../data",
    )
    args = parser.parse_args()
    prepare(pkl_path=args.pkl_path)
