"""
Unit tests for model.py
Requires a sample API json for testing

"""

import pandas as pd

from frontend.model import leaguedash_columns, model


def test_load_pickle(league_pickle):
    """Tests for loading json pickle into dataframe"""
    res = model.load_pickle(league_pickle)
    assert isinstance(res, pd.DataFrame)


def test_feature_engineer(league_raw_df):
    """Tests for correct removal of columns from leaguedash df
    Takes as input the result of load_pickle"""
    res = model.feature_engineer(league_raw_df)
    res_cols = set(res.columns)
    assert isinstance(res, pd.DataFrame)
    assert set(["FG2M", "FG2A"]) <= res_cols  # subset
    assert not set(leaguedash_columns.DROP_COLS) & res_cols  # no intersection


def test_transform_leaguedash(league_raw_df):
    """Tests for correct transformation"""
    res = model.transform_leaguedash(reg_df=league_raw_df, post_df=league_raw_df)
    assert isinstance(res, pd.DataFrame)
