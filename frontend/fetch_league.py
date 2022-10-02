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

# def get_leaguedash():
#     """Calls NBA API for league dash player stats endpoint"""
#     league_dash = leaguedashplayerstats.LeagueDashPlayerStats(

#     )
