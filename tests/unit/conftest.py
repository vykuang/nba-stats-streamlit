import pickle
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def test_data_dir() -> Path:
    """Returns the test data path"""
    return Path(__file__).parents[1] / "data"


@pytest.fixture
def make_league_pickle(test_data_dir) -> Any:
    """Factory fixture to create file path for both
    regular and playoffs pickle
    """

    def _make_league_pickle(season_type: str) -> Path:
        """Loads and returns pickle based on input
        Accepts {"regular", "playoffs"}
        """
        return test_data_dir / f"leaguedash_{season_type}_2018-19.pkl"

    return _make_league_pickle


@pytest.fixture
def make_league_df(test_data_dir) -> Any:
    """Factory fixture
    Returns a function which returns the pre-pickled results of the leaguedash
    dataframe based on argument
    Accepts {"regular", "playoffs", "merge"}
    """

    def _make_league_df(season_type: str) -> pd.DataFrame:
        """Loads and returns dataframe based on string input"""
        df_path = test_data_dir / f"test_{season_type}_df.pkl"
        with open(df_path, "rb") as pkl:
            return pickle.load(pkl)

    return _make_league_df
