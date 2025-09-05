"""
leaguedash retrieves the season's cumulative stat for all players
serves as the historical data used to train the clustering model
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from minio import Minio
from nba_api.stats.endpoints import leaguedashplayerstats

load_dotenv()

MINIO_ROOT_USER = os.getenv('MINIO_ROOT_USER')
MINIO_ROOT_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')
MINIO_URL = os.getenv('MINIO_URL', "http://localhost:9001")

storage_opt = {
        "key": MINIO_ROOT_USER,
        "secret": MINIO_ROOT_PASSWORD,
        "client_kwargs": {"endpoint_url": MINIO_URL}
    }
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)



def minio_file_exists(s3_uri: str) -> bool:
    """
    Checks if a file exists in MinIO storage given an S3 URI.
    Example S3 URI: s3://bucket/file.parquet
    """
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    object_name = parsed.path.lstrip('/')

    client = Minio(
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        endpoint=MINIO_URL.replace("http://", "").replace("https://", ""),
        secure=MINIO_URL.startswith("https://"),
    )
    return client.bucket_exists(bucket) and client.stat_object(bucket, object_name) is not None


def get_leaguedash_json(
    season_type: str,
    season: str,
    measure_type: str = "Base",
) -> pd.DataFrame:
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
    # the method returns a list of df
    # leaguedash is just one df, so we just retrieve the first and only df
    return league_dash.get_data_frames()[0] 


def fetch_league_dash(
    season: str = "2020-21",
    data_path: str = "s3://data/raw",
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
    reg_path = f"{data_path}/leaguedash_regular_{season}.parquet"
    playoffs_path = f"{data_path}/leaguedash_playoffs_{season}.parquet"
    logger.debug(
        f"""
        Saving to:
        Regular season: {reg_path}
        Playoffs: {playoffs_path}
        """
    )

    if not minio_file_exists(reg_path):
        logger.info(f"Retrieving regular season dashboard for season {season}")
        regular = get_leaguedash_json("Regular Season", season)
        logger.debug(f"Num of records retrieved: {len(regular)}")
        logger.info(f"Saving regular season results to:\n{reg_path}")
        regular.to_parquet(reg_path, storage_options=storage_opt)
        # so we don't get blocked from API requests
        wait_time = random.gammavariate(alpha=9.0, beta=0.4)
        logger.debug(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
    else:
        logger.info(f"{reg_path} already exists; API not called")

    if not minio_file_exists(playoffs_path):
        logger.info(f"Retrieving playoff dashboard for {season}")
        playoffs = get_leaguedash_json("Playoffs", season)
        logger.debug(f"Num of records retrieved: {len(playoffs)}")
        logger.info(f"Saving playoffs results to:\n{playoffs_path}")
        playoffs.to_parquet(playoffs_path, storage_options=storage_opt)
    else:
        logger.info(f"{playoffs_path} already exists; API not called")

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
        default="s3://data/raw",
    )
    parser.add_argument(
        "--loglevel",
        "-l",
        type=str.upper,
        default="INFO",
    )

    args = parser.parse_args()
    main(args.season, args.data_path, args.loglevel)
