import pickle
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def test_data_dir() -> Path:
    """Returns the test data path"""
    return Path(__file__).parents[1] / "data"


@pytest.fixture
def make_league_pickle(test_data_dir):
    """Factory fixture to create file path for both
    regular and playoffs pickle
    """

    def _make_league_pickle(season_type: str = "regular") -> Path:
        """Loads and returns pickle based on input
        Accepts {"regular", "playoffs"}
        """
        return test_data_dir / f"leaguedash_{season_type}_2018-19.pkl"

    return _make_league_pickle


def load_pickle(fp: Path):
    """Loads the pickle"""
    with open(fp, "rb") as f_in:
        res = pickle.load(f_in)

    return res


@pytest.fixture
def make_league_df(test_data_dir):
    """Factory fixture
    Returns a function which returns the pre-pickled results of the leaguedash
    dataframe based on argument
    Used to test load_pickle and feature_engineer
    """

    def _make_league_df(season_type: str = "regular") -> pd.DataFrame:
        """Loads and returns dataframe based on string input
        Accepts {"regular", "playoffs", "merge", "rerank", "transform"}"""
        df_path = test_data_dir / f"test_{season_type}_df.pkl"
        return load_pickle(df_path)

    return _make_league_df


@pytest.fixture
def make_feat_df(test_data_dir):
    """
    Returns a function that can return either the feature engineered
    regular or post df
    Tests feature_engineer results
    """

    def _make_feat_df(season_type: str = "regular") -> pd.DataFrame:
        """
        Parameters
        ---------

        season_type: str, {"regular", "playoffs"}

        Returns
        --------

        feat_df: pd.DataFrame
            league df that's been feature engineered and known
            to be correct
        """
        fp = test_data_dir / f"test_{season_type}_feat.pkl"
        return load_pickle(fp)

    return _make_feat_df
