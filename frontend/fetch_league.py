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
from nba_api.stats.endpoints import leaguedashplayerstats


def get_leaguedash_json(
    season_type: str,
    season: str,
    measure_type: str = "Base",
):
    """Calls stats.nba.com/stats/leaguedashplayerstats

    Parameters
    ----------
    season_type: {'Regular Season', 'Playoffs'}
        Which season segment to request stats for

    season: str
        Format as YYYY-YY, e.g. 2020-21

    measure_type: {'Base', 'Advanced'}, default 'Base'

    Returns
    -------
    league_dash: dict
        json of the API result
    """
    league_dash = leaguedashplayerstats.LeagueDashPlayerStats(
        measure_type_detailed_defense=measure_type,
        season_type_all_star=season_type,
        season=season,
        plus_minus="N",
        per_mode_detailed="Per36",
    )
    return league_dash.get_normalized_json()


def weighted_stats(regular: dict, playoffs: dict, post_wt: float = 2.0) -> dict:
    """Merge stats proportionally via games played"""


def dump_pickle(obj, fp: Path) -> None:
    """pickle dump"""
    with open(fp, "wb") as f_out:
        pickle.dump(obj, f_out)


def fetch_league_dash(
    season: str = "2020-21",
    data_path: Path = Path("../data"),
) -> None:
    """Calls stats.nba.com/stats/leaguedashplayerstats for the specified season,
    requesting both regular season and playoffs stats. Stores results as a pickle
    in data_path

    Parameters
    ----------
    season: str
        Format as YYYY-YY, e.g. 2020-21

    data_path: Path
        location to store the API call results

    Returns
    -------
    None
    """
    regular = get_leaguedash_json("Regular Season", season)
    playoffs = get_leaguedash_json("Playoffs", season)
    season = weighted_stats(regular, playoffs)
    dump_pickle(season, data_path / "leaguedash.pkl")


def _run(season: str, data_path: Path) -> None:
    """"""
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=False)

    fetch_league_dash(season, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Fetch NBA league dashboard")
    parser.add_argument(
        "--season",
        "-s",
        type=str,
        default="2018-2019",
    )
    parser.add_argument(
        "--data_path",
        "-p",
        type=Path,
        default="../data",
    )

    args = parser.parse_args()
    _run(args.season, args.data_path)
