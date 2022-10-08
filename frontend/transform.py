"""
Preprocesses the downloaded data from NBA API calls
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)

ALL_COLS = [
    "PLAYER_NAME",
    "NICKNAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "AGE",
    "GP",
    "W",
    "L",
    "W_PCT",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
    "WNBA_FANTASY_PTS",
    "GP_RANK",
    "W_RANK",
    "L_RANK",
    "W_PCT_RANK",
    "MIN_RANK",
    "FGM_RANK",
    "FGA_RANK",
    "FG_PCT_RANK",
    "FG3M_RANK",
    "FG3A_RANK",
    "FG3_PCT_RANK",
    "FTM_RANK",
    "FTA_RANK",
    "FT_PCT_RANK",
    "OREB_RANK",
    "DREB_RANK",
    "REB_RANK",
    "AST_RANK",
    "TOV_RANK",
    "STL_RANK",
    "BLK_RANK",
    "BLKA_RANK",
    "PF_RANK",
    "PFD_RANK",
    "PTS_RANK",
    "PLUS_MINUS_RANK",
    "NBA_FANTASY_PTS_RANK",
    "DD2_RANK",
    "TD3_RANK",
    "WNBA_FANTASY_PTS_RANK",
    "CFID",
    "CFPARAMS",
]

DROP_STATS = [
    "NICKNAME",
    "TEAM_ID",
    "W",
    "L",
    "FGM",
    "FGA",
    "REB",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
    "WNBA_FANTASY_PTS",
    "CFID",
    "CFPARAMS",
]
DROP_RANK_PCT = [col for col in ALL_COLS if "_RANK" in col or "_PCT" in col]
DROP_COLS = DROP_STATS + DROP_RANK_PCT


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Removes extraneous columns from leaguedash,
    and engineers some new features
    """
    result = df.copy()
    result["FG2M"] = result["FGM"] - result["FG3M"]
    result["FG2A"] = result["FGA"] - result["FG3A"]

    result = result.drop(DROP_COLS, axis=1)
    return result


def reg_post_merge(reg_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    """Folds regular and post season stats into one via a weight coefficient"""
    # if either regular or post stats for a given player is missing, use
    # what's present
    # only fold if both are present


def player_meets_standard(player: pd.Series) -> bool:
    """Does this player have >= 500 min or >= 40 games played?
    Considers the folded minutes/games played
    """


def transform_leaguedash(reg_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares API results for clustering"""
    # feature engineer
    reg_df = feature_engineer(reg_df)
    post_df = feature_engineer(post_df)

    # merge
    merge_df = reg_df.apply(lambda player: reg_post_merge(player), axis=1)

    # filter for minutes and games played
    player_filter = merge_df.apply(player_meets_standard, axis=1)
    return merge_df[player_filter]


def dump_pickle(obj, fp: Path) -> None:
    """pickle dump"""
    with open(fp, "wb") as f_out:
        pickle.dump(obj, f_out)


def load_pickle(fp: Path) -> Any:
    """Loads the json pickle and returns as df"""
    with open(fp, "rb") as f_in:
        res = pickle.load(f_in)

    if isinstance(res, list) and isinstance(res[0], dict):
        df = pd.DataFrame.from_dict(res).set_index("PLAYER_ID")
        return df
    else:
        raise TypeError("Expected list of dicts")


def transform(season: str, data_path: Path, loglevel: str):
    """"""
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logger.setLevel(numeric_level)
    handler.setLevel(numeric_level)
    logger.addHandler(handler)

    reg_pkl = data_path / f"leaguedash_regular_{season}.pkl"
    post_pkl = data_path / f"leaguedash_playoffs_{season}.pkl"
    merge_pkl = data_path / f"leagedash_merge_{season}.pkl"
    reg_df = load_pickle(reg_pkl)
    post_df = load_pickle(post_pkl)
    merge_df = transform_leaguedash(reg_df, post_df)
    dump_pickle(merge_df, merge_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Transforms nba leaguedash stats")
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
        "--loglevel",
        "-l",
        type=str.upper,
        default="INFO",
    )

    args = parser.parse_args()
    transform(args.season, args.data_path, args.loglevel)
