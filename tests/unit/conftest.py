import pytest


@pytest.fixture
def make_feat_df(test_data_dir, load_pickle):
    """
    Returns a function that can return either the feature engineered
    regular or post df
    Tests feature_engineer results
    """

    def _make_feat_df(season_type: str = "regular"):
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
