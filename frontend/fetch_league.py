import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players
from tqdm import tqdm


def get_leaguedash(
    measure_type: str = "Base",
    season_type: str = "Regular Season",
    season: str = "2001-02",
    df: bool = True,
):
    """Calls NBA API for league dash player stats endpoint"""
    league_dash = leaguedashplayerstats.LeagueDashPlayerStats(
        measure_type_detailed_defense=measure_type,
        season_type_all_star=season_type,
        season=season,
        plus_minus="Y",
        per_mode_detailed="Per100Possessions",
    )
    if df:
        res = league_dash.get_data_frames()[0]
    else:
        res = json.loads(league_dash.get_normalized_json())

    return res
