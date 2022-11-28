import pickle
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parents[1] / "data"


# @pytest.fixture
# def reg_pickle(test_data_dir) -> Tuple[Path, Path]:
#     return test_data_dir / "leaguedash_regular_2018-19.pkl"


# @pytest.fixture
# def post_pickle(test_data_dir) -> Path:
#     """
#     """
#     return test_data_dir / "leaguedash_playoffs_2018-19.pkl"


@pytest.fixture
def make_league_pickle(test_data_dir) -> Path:
    def _make_league_pickle(regular: bool = True):
        if regular:
            season_type = "regular"
        else:
            season_type = "playoffs"
        return test_data_dir / f"leaguedash_{season_type}_2018-19.pkl"

    return _make_league_pickle


@pytest.fixture
def make_league_df(test_data_dir) -> pd.DataFrame:
    """
    Returns the pre-pickled results of the leaguedash dataframe
    """

    def _make_league_df(regular: bool = True):
        if regular:
            season_type = "regular"
        else:
            season_type = "playoffs"
        df_path = test_data_dir / f"test_{season_type}_df.pkl"
        with open(df_path, "rb") as pkl:
            return pickle.load(pkl)

    return _make_league_df
