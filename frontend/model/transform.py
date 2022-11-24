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

PLAYER_BIO = set(["PLAYER_NAME", "TEAM_ABBREVIATION", "AGE"])
MERGE_STATS = [
    "GP",
    "MIN",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
    "FG2M",
    "FG2A",
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


def transform_leaguedash(
    reg_df: pd.DataFrame, post_df: pd.DataFrame, post_wt: float = 2.0
) -> pd.DataFrame:
    """Prepares API results for clustering"""
    # feature engineer
    logger.debug("feature engineering...")
    reg_df = feature_engineer(reg_df)
    post_df = feature_engineer(post_df)
    logger.info("feature engineering complete")

    post_ids = set(post_df.index)
    # merge
    def reg_post_merge(player: pd.Series, post_wt: float = 2.0) -> pd.Series:
        """Folds regular and post season stats into one via a weight coefficient"""
        # if either regular or post stats for a given player is missing, use
        # what's present
        # only fold if both are present
        player = player.copy()  # avoids mutating the df as it's being iterated
        if player.name in post_ids:
            post_season = post_df.loc[player.name]
            # initiate merge, since player is present in both reg and post

        else:
            post_season = player

        gp_tot = player["GP"] + post_wt * post_season["GP"]
        for stat in player.index:
            if stat not in PLAYER_BIO:
                player[stat + "_merge"] = (
                    player["GP"] / gp_tot * player[stat]
                    + post_wt * post_season["GP"] / gp_tot * post_season[stat]
                )
        return player

    logger.debug("Merging regular and post season stats...")
    # drop reg season stats after merging with post season
    merge_df = reg_df.apply(reg_post_merge, post_wt=post_wt, axis=1).drop(
        MERGE_STATS, axis=1
    )
    logger.info(f"Merging complete with post_wt = {post_wt:.3f}")
    logger.debug(f"Players post merge: {len(merge_df)}")

    # re-rank using merged stats
    def leaguedash_rerank(stat: pd.Series) -> pd.Series:
        """Ranks all the values in the given stat column.
        Largest values will be given top ranks
        To be used in df.apply()

        Parameters
        ---------

        stat: pd.Series
            A statistical field with numeric values to be ranked

        Returns
        --------

        stat_rank: pd.Series
            Ranking of the stat Series
        """

        # sort the values
        sorted_stat_index = stat.sort_values(ascending=False).index

        # attach a sequential index to the now sorted values
        sorted_rank = [(rank + 1) for rank in range(len(stat.index))]

        # can't for the life of me figure out how to return my desired column names
        # rename after returning.
        rank_series = pd.Series(
            data=sorted_rank,
            index=sorted_stat_index,
            name=f"{stat.name}_RANK",
        ).reindex(index=stat.index)

        # standardize by dividing by num of players
        rank_series /= len(stat)
        return rank_series

    logger.debug("Re-ranking merged stats...")
    # only rank merged columns, so drop bio before merging
    merge_ranks = merge_df.drop(PLAYER_BIO, axis=1).apply(
        leaguedash_rerank, axis="index"
    )
    merge_ranks.columns = [col.replace("merge", "RANK") for col in merge_ranks.columns]
    logger.info("Re-rank complete")

    logger.debug(
        f"merge_df shape: {merge_df.shape}\nmerge_rank shape: {merge_ranks.shape}"
    )
    merge_df = pd.concat([merge_df, merge_ranks], axis="columns")
    logger.info("Ranks and stats merge complete")
    logger.debug(f"Column count: {len(merge_df.columns)}")

    # filter for minutes and games played
    def player_meets_standard(
        player: pd.Series, min_thd: int = 800, gp_thd: int = 40
    ) -> bool:
        """Does this player pass the minutes or games played threshold?
        Considers the folded minutes/games played
        """
        return player["MIN_merge"] >= min_thd or player["GP_merge"] >= gp_thd

    merge_df["gametime_threshold"] = merge_df.apply(player_meets_standard, axis=1)
    logger.info(f"Number of eligible players: {merge_df['gametime_threshold'].sum()}")

    return merge_df


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


def transform(
    season: str, data_path: Path, loglevel: str = "info", overwrite: bool = False
):
    """Loads the pickle for transformation, and stores the result"""
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logger.setLevel(numeric_level)
    handler.setLevel(numeric_level)
    logger.addHandler(handler)

    reg_pkl = data_path / f"leaguedash_regular_{season}.pkl"
    post_pkl = data_path / f"leaguedash_playoffs_{season}.pkl"
    merge_pkl = data_path / f"leaguedash_merge_{season}.pkl"
    if merge_pkl.exists() and not overwrite:
        logger.info(
            f"""
            Merged pickle already exists at:
            {merge_pkl.resolve()}
            Exiting
            """
        )
        return True

    logger.info(f"Loading from {data_path.resolve()}")
    reg_df = load_pickle(reg_pkl)
    post_df = load_pickle(post_pkl)

    logger.debug("Pickles loaded")
    logger.debug(f"Loaded {len(reg_df)} records from reg_pkl")
    logger.debug(f"Loaded {len(post_df)} records from post_pkl")

    merge_df = transform_leaguedash(reg_df, post_df)

    dump_pickle(merge_df, merge_pkl)
    logger.info(f"Results saved to {merge_pkl.resolve()}")
    return False


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
    parser.add_argument(
        "--overwrite",
        "-o",
        type=bool,
        default=False,
        help="If True, overwrite existing transform result pickle",
    )
    args = parser.parse_args()
    transform(args.season, args.data_path, args.loglevel, args.overwrite)
