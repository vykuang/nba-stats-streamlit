"""
Unit tests for model.py
Requires a sample API json for testing

"""

import pandas as pd

from frontend.model import leaguedash_columns, model


def test_load_pickle(make_league_pickle, make_league_df):
    """Tests for loading json pickle into dataframe"""
    # setup
    reg_pkl = make_league_pickle("regular")
    post_pkl = make_league_pickle("playoffs")

    test_reg_pkl = make_league_df("regular")
    test_post_pkl = make_league_df("playoffs")
    # testing
    reg_pkl_res = model.load_pickle(reg_pkl)
    post_pkl_res = model.load_pickle(post_pkl)

    # simply using `==` actually returns another df
    # use .all() to confirm that the resulting df is all True
    # specify axis; otherwise still returns series/df
    assert (reg_pkl_res == test_reg_pkl).all(axis=None)
    assert (post_pkl_res == test_post_pkl).all(axis=None)


def test_feature_engineer(make_league_df):
    """Tests for correct removal of columns from leaguedash df
    Takes as input the result of load_pickle"""
    res = model.feature_engineer(make_league_df)
    res_cols = set(res.columns)
    assert isinstance(res, pd.DataFrame)
    assert set(["FG2M", "FG2A"]) <= res_cols  # subset
    assert not set(leaguedash_columns.DROP_COLS) & res_cols  # no intersection


def test_reg_post_merge():
    """
    Tests the merging of regular and playoffs leaguedash df
    Used in reg_df.apply()
    """
    # setup
    test_reg_df = make_league_df("regular")
    test_post_df = make_league_df("playoffs")
    test_res = make_league_df("merge")


def test_rerank():
    """
    Tests the ranking func used in merge_df.apply()
    """


def test_meet_standard():
    """
    Tests whether the player filter criteria is correctly judged
    """


def test_transform_leaguedash(make_league_df):
    """Tests for correct transformation
    Acts as integration test for reg_post_merge, rerank, and meets_standard
    """
    res = model.transform_leaguedash(
        reg_df=make_league_df("regular"), post_df=make_league_df("playoffs")
    )

    assert isinstance(res, pd.DataFrame)
    assert 0
