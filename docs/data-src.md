# Data Ingestion

Data is sourced from NBA's stats API. To facilitate this, `nba_api` will be imported so python can programmatically fetch and store the requested json data inside a postgres container

## Endpoints

NBA's stats API offer a lot of endpoints. Since we're interested in a player's season averages, we are looking at `cumestatsplayer` endpoint, and restricting it to all games within 21-22 season.

Couldn't get it to work; unable to pass multiple game_IDs to fetch the stats. The composed request URL contains all the IDs, but only the first one is accepted.

`playercareerstats` used instead.

To gather the player_ids, use the `static` endpoint to gather all active players.

## Pipeline

1. Gather all active player_ids via static endpoint
1. For each player_id, request `playercareerstats` API.
1. Check that `MIN` played > 500 minutes, or 'GP' > 40; remove others
1. Fold the SeasonTotalsRegular and SeasonTotalsPost to a single dict
1. Store all player's data into .json for later storage into postgres

## Execution

- Need to add a progress bar and log DEBUG output (what player's stats am I requesting?)

- Switch to `.csv` for easy append so that I can save my progress if my API calls are blocked:

  ```py
  import csv
  csv_path = '../data/foo.csv'
  fields = ['a', 'b', 'c']
  # 'a' for append
  with open(csv_path, 'a') as f:
      write = csv.writer(f)
      write.writerow(fields)
  ```

  - Maybe this isn't as straightforward. For one thing, there's a lot of commas in JSON. For two, csv is just not as expressive. We've got list of dicts of lists of dicts. It's all nested. Ideally this would be flattened before storing as csv.
  - What if we just stored it as native json?

- use `tqdm` as recommended in python for machine learning to output a progress bar for fetching the 500+ API calls

  ```py
  from tqdm import tqdm
  for player in tqdm(players):
      stat = fetch_stat(player)

  """
  %|▌              | 3/587 [00:08<28:52,  2.97s/it]
  """
  ```

- Handle players that may not have played in either post or regular season

## Data cleaning

Originally did *some* data cleaning in the fetching stage. I'm beginning to think that all of it should be left alone, and any transformations should wait until the actual data cleaning and feature engineering stage.

When compiling, I thought of dropping some features which are by design collinear, but:

- I missed one, so I had to drop it in another data cleaning phase
- more feature engineering had to be done anyway, so data cleaning wasn't optional
