import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def get_career_stats(player_id: str, get_request: bool = True) -> dict:
    """Calls NBA API for playercareerstats"""
    career_stats = playercareerstats.PlayerCareerStats(
        player_id=player_id,
        per_mode36="Per36",
        get_request=get_request,
    )
    return json.loads(career_stats.get_normalized_json())


def get_jsons(career_stats: dict) -> Tuple[dict, dict]:
    """Takes in playercareeer dataset as dict and returns
    the regular and post season totals, for the most recent season"""
    logger = logging.getLogger("get_jsons")
    logger.debug(f"{career_stats}")
    reg_season = []
    post_season = []
    if career_stats["SeasonTotalsRegularSeason"]:
        reg_season = career_stats["SeasonTotalsRegularSeason"][-1]

    if career_stats["SeasonTotalsPostSeason"]:
        post_season = career_stats["SeasonTotalsPostSeason"][-1]

    return (reg_season, post_season)


def player_meets_standard(
    reg: dict, post: dict, min_thd: int = 500, gp_thd: int = 40
) -> bool:
    """Does this player have >= 500 min or >= 40 games played?"""
    if reg and post:
        meets_standard = bool(
            reg["MIN"] + post["MIN"] >= min_thd or reg["GP"] + post["GP"] >= gp_thd
        )
    elif reg and not post:
        meets_standard = bool(reg["MIN"] >= min_thd or reg["GP"] >= gp_thd)
    elif not reg and post:
        meets_standard = bool(post["MIN"] >= min_thd or post["GP"] >= gp_thd)
    else:
        meets_standard = False
    return meets_standard


def fold_post_stats(
    reg_season: dict, post_season: dict, merge_stats: list, post_wt: float = 2.0
) -> dict:
    """Merge stats proportionally via games played"""
    merged = {}
    if player_meets_standard(reg_season, post_season):

        if reg_season and post_season:

            gp_tot = reg_season["GP"] + post_wt * post_season["GP"]
            for stat in merge_stats:
                merged[stat] = (
                    reg_season["GP"] / gp_tot * reg_season[stat]
                    + post_wt * post_season["GP"] / gp_tot * post_season[stat]
                )

            merged["PLAYER_ID"] = reg_season["PLAYER_ID"]
            # merged['SEASON_ID'] = reg_season['SEASON_ID']
        elif reg_season and not post_season:
            merged = {stat: reg_season[stat] for stat in merge_stats}
            merged["PLAYER_ID"] = reg_season["PLAYER_ID"]
        else:
            merged = {stat: post_season[stat] for stat in merge_stats}
            merged["PLAYER_ID"] = post_season["PLAYER_ID"]
    return merged


def dump_pickle(obj, fp: Path) -> None:
    """pickle dump"""
    with open(fp, "wb") as f_out:
        pickle.dump(obj, f_out)


def load_pickle(fp: Path) -> Any:
    """pickle load"""
    with open(fp, "rb") as f_in:
        return pickle.load(f_in)


def append_json(json_path: Path, new_data: dict, update_key: str = None) -> None:
    """Loading and updating dict to the json"""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(json_path)
        if update_key:
            data[update_key].append(new_data)
        else:
            data.update(new_data)

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def fetch(pkl_path: Path = Path("../data")) -> None:
    """Calls stats.nba API for each active player and stages
    in local dir
    This is separate from preprocessing since API calls are an
    expensive resource and should not be retried often
    """
    stats_path = Path(pkl_path / "careerstats.pkl")
    # don't call API if the pkl already exists
    if stats_path.exists():
        return

    json_path = Path(pkl_path / "careerstats.json")

    active = players.get_active_players()
    player_map_path = Path(pkl_path / "player_map.pkl")
    player_dict = {}
    for player in tqdm(active):
        # build map to find player name
        player_dict[player["id"]] = player["full_name"]
        # progress logging
        # logging.info(f'\rProgress: {idx/num_players:.2%}\tRequesting stat for {player["full_name"]}', end="\r")
        # calls stats.nba API for each active player
        career_stats = get_career_stats(player["id"])
        append_json(json_path, {player["id"]: career_stats})

        # avoids hammering the API and being blocked
        wait = random.gammavariate(alpha=9.0, beta=0.4)
        time.sleep(wait)

    with open(json_path, "r", encoding="utf-8") as file:
        active_stats = json.load(file)

    dump_pickle(active_stats, stats_path)
    dump_pickle(player_dict, player_map_path)


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
