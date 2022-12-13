import pickle
from pathlib import Path

import pandas as pd
import pytest

from frontend.model import model


@pytest.fixture
def test_data_dir() -> Path:
    """Returns the test data path"""
    return Path(__file__).parent / "data"


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


@pytest.fixture
def load_pickle():
    def _load_pickle(fp: Path):
        """Loads the pickle"""
        with open(fp, "rb") as f_in:
            res = pickle.load(f_in)

        return res

    return _load_pickle


@pytest.fixture
def make_league_df(test_data_dir, load_pickle):
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
def mock_env_mlflow(monkeypatch, tmp_path):
    """Sets MLflow env vars to test state

    By the time this fixture runs, model.py will already have tried to
    retrieve the environment vars
    This sets the module constants after the env vars have been retrieved
    """
    # need to set a filepath so that model artifact are also saved im tmp_path
    # instead of local project directory
    monkeypatch.setattr(model, "MLFLOW_TRACKING_URI", f"{str(tmp_path / 'mlruns')}")
    monkeypatch.setattr(
        model, "MLFLOW_REGISTRY_URI", f"sqlite:///{str(tmp_path / 'mlflow.db')}"
    )
    monkeypatch.setattr(model, "MLFLOW_EXP_NAME", "pytest")
    monkeypatch.setattr(model, "MLFLOW_REGISTERED_MODEL", "pytest-clusterer")
    monkeypatch.setattr(model, "MLFLOW_ARTIFACT_PATH", "pytest-model")
