"""
Preprocesses the downloaded data from NBA API calls
"""
import logging
import pickle
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)


def player_meets_standard():
    """Does this player have >= 500 min or >= 40 games played?"""
