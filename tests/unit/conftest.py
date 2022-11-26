from pathlib import Path

import pandas as pd
import pytest

from frontend.model import model


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parents[1] / "data"


@pytest.fixture
def league_pickle(test_data_dir) -> Path:
    return test_data_dir / "leaguedash_playoffs_2015-16.pkl"


@pytest.fixture
def league_raw_df(league_pickle) -> pd.DataFrame:
    """
    I do see the paradox in using a function I'm supposed to be testing
    in my test suite setup
    Perhaps this calls for some slight restructuring
    """
    return model.load_pickle(league_pickle)
