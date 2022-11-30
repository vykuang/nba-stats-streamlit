"""
Unit tests for model.py
Requires sample API json for testing
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


def test_feature_engineer(make_league_df, make_feat_df):
    """Tests for correct removal of columns from leaguedash df
    Takes as input the result of load_pickle"""
    # setup
    test_feat = make_feat_df()

    # execute
    res = model.feature_engineer(make_league_df())
    res_cols = set(res.columns)

    # assert
    assert isinstance(res, pd.DataFrame)
    assert set(["FG2M", "FG2A"]) <= res_cols  # subset
    assert not set(leaguedash_columns.DROP_COLS) & res_cols  # no intersection
    assert (res == test_feat).all(axis=None)


def test_reg_post_merge(make_league_df, make_feat_df):
    """
    Tests the merging of regular and playoffs leaguedash df
    Used in reg_df.apply()
    """
    # setup
    test_reg_df = make_feat_df("regular")
    test_post_df = make_feat_df("playoffs")
    # ground truth
    test_merge_df = make_league_df("merge")

    # test
    # merge_res = test_reg_df.apply(
    #     model.reg_post_merge,
    #     post_df=test_post_df,
    #     axis=1
    # ).drop(leaguedash_columns.MERGE_STATS, axis=1)
    merge_res = model.reg_post_merge(test_reg_df, test_post_df)

    assert (merge_res == test_merge_df).all(axis=None)


def test_rerank(make_league_df):
    """
    Tests the ranking func used in merge_df.apply()
    """
    # setup
    test_rerank_df = make_league_df("rerank")
    test_merge_df = make_league_df("merge")
    # execute
    res = model.leaguedash_rerank(test_merge_df)

    # assert
    assert (res == test_rerank_df).all(axis=None)


def test_player_standard(make_league_df):
    """Tests the player check function"""
    # setup
    test_rerank_df = make_league_df("rerank")
    test_gametime_df = make_league_df("gametime")
    num = 20
    player = {"MIN_merge": num, "GP_merge": num}

    # execute
    res_gametime = model.player_meets_standard(test_rerank_df)
    # pylint: disable=W0212
    # both pass
    assert model._player_meets_standard(player=player, min_thd=num, gp_thd=num)
    # only GP pass
    assert model._player_meets_standard(player=player, min_thd=num + 1, gp_thd=num)
    # only MIN pass
    assert model._player_meets_standard(player=player, min_thd=num, gp_thd=num + 1)
    # neither pass
    assert not model._player_meets_standard(
        player=player, min_thd=num + 1, gp_thd=num + 1
    )
    # test whole df
    # pylint: enable=W0212
    assert (res_gametime == test_gametime_df).all(axis=None)


def test_transform_leaguedash(make_league_df):
    """Tests for correct transformation
    Acts as integration test for reg_post_merge, rerank, and meets_standard
    """
    test_transform_df = make_league_df("gametime")
    res = model.transform_leaguedash(
        reg_df=make_league_df("regular"), post_df=make_league_df("playoffs")
    )

    # assert
    assert isinstance(res, pd.DataFrame)
    assert (res == test_transform_df).all(axis=None)
