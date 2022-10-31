#! usr/bin/env python
import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguedashplayerstats

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)


def get_leaguedash_json(
    season_type: str,
    season: str,
    measure_type: str = "Base",
) -> list:
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
    league_dash: list
        API result encapsulated as list of player records, each
        record a dict with {'stat_a': val_a, 'stat_b': val_b, ...}
    """
    league_dash = leaguedashplayerstats.LeagueDashPlayerStats(
        measure_type_detailed_defense=measure_type,
        season_type_all_star=season_type,
        season=season,
        plus_minus="N",
        per_mode_detailed="Per36",
    )
    res = league_dash.get_normalized_json()
    return json.loads(res)["LeagueDashPlayerStats"]


# def weighted_stats(regular: dict, playoffs: dict, post_wt: float = 2.0) -> dict:
#     """Merge stats proportionally via games played

#     Parameters
#     ----------
#     regular: dict
#         """


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
    reg_path = data_path / f"leaguedash_regular_{season}.pkl"
    playoffs_path = data_path / f"leaguedash_playoffs_{season}.pkl"
    logger.debug(
        f"""
        Saving to:
        Regular season: {reg_path.resolve()}
        Playoffs: {playoffs_path.resolve()}
        """
    )

    if not reg_path.exists():
        logger.info(f"Retrieving regular season dashboard for season {season}")
        regular = get_leaguedash_json("Regular Season", season)
        logger.debug(f"Num of records retrieved: {len(regular)}")
        logger.info(f"Saving regular season results to:\n{reg_path.resolve()}")
        dump_pickle(regular, reg_path)

        # so we don't get blocked from API requests
        wait_time = random.gammavariate(alpha=9.0, beta=0.4)
        logger.debug(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
    else:
        logger.info(f"{reg_path.name} already exits; API not called")

    if not playoffs_path.exists():
        logger.info(f"Retrieving playoff dashboard for {season}")
        playoffs = get_leaguedash_json("Playoffs", season)
        logger.debug(f"Num of records retrieved: {len(playoffs)}")
        logger.info(f"Saving playoffs results to:\n{playoffs_path.resolve()}")
        dump_pickle(playoffs, playoffs_path)
    else:
        logger.info(f"{playoffs_path.name} already exits; API not called")


def main(season, data_path, loglevel):
    """
    Wrapper for fetch_league_dash to parametrize logging level
    """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logger.setLevel(numeric_level)
    handler.setLevel(numeric_level)
    logger.addHandler(handler)

    fetch_league_dash(season=season, data_path=data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Fetch NBA league dashboard")
    parser.add_argument(
        "--season",
        "-s",
        type=str,
        default="2018-19",
    )
    parser.add_argument(
        "--data_path",
        "-p",
        type=Path,
        default="../data",
    )
    parser.add_argument(
        "--dryrun",
        "-d",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--loglevel",
        "-l",
        type=str.upper,
        default="INFO",
    )

    args = parser.parse_args()

    if not args.data_path.exists():
        args.data_path.mkdir(parents=True, exist_ok=False)
    if not args.dryrun:
        main(args.season, args.data_path, args.loglevel)
