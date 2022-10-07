# Data Ingestion

Data is sourced from NBA's stats API. To facilitate this, `nba_api` will be imported so python can programmatically fetch and store the requested json data inside a postgres container

## Endpoints

NBA's stats API offer a lot of endpoints. Since we're interested in a player's season averages, we are looking at `cumestatsplayer` endpoint, and restricting it to all games within 21-22 season.

Couldn't get it to work; unable to pass multiple game_IDs to fetch the stats. The composed request URL contains all the IDs, but only the first one is accepted.

`playercareerstats` used instead.

To gather the player_ids, use the `static` endpoint to gather all active players.

### UPDATE

Per [this post on SOF](https://stackoverflow.com/a/71556033/5496416) we can view what exactly the request header was when we make a request via nba.com console, and through that, I found out that to pass through multiple game_ids, they must be delimited with `|`, instead of the default `&`.

I need to find the source code where they compose the request header and modify the `&` to `|`.

Source code doesn't seem to do any of that in the first place. The `&` comes from the list being passed to it. Perhaps I can just pass a single string where the IDs are delimited by `|`, instead of a list.

Breakthrough here is definitely this snippet:

> You find the endpoint by opening Dev Tools (shift-ctrl-i) and look under Network -> XHR (you may need to refresh the page). Watch the panel for the requests to start popping up, and find the one that has your data. Go to Headers to find the info needed to make the request

### Differentiating by season

If I want to compare across eras, I need a way to delineate the seasons. My current workflow simply retrieves all active players from the present, but of course that leaves out all retired players, so how would I retrieve active players for each season? Perhaps some combination of `teamgamelog`?

`leaguedashplayerstats` seems to be a kind of catch-all endpoint for all the common stat, and allows as to filter by season and regular/post for their aggregated stats. Honestly this looks exactly like what I need, in only one API call as well. Or two, if you count regular plus post season. We could also extend it by including `Advanced` stats as well, by changing `MeasureType` param from `Base` to `Advanced`

If I can get a whole season's worth in one go it makes more sense to get multiple seasons

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

### Incremental writes

In the interest of saving data, and possibly picking up where API requests were interrupted, program should write results to disc after each successful call.

Most straightforward way is the following

1. Call API request
1. Open .json
1. Load dict from .json
1. Update dict with API result
1. Save updated dict in .json

The *inelegance* of it is that we're repeating the file I/O each time we make the call. The reality of is this just works.

### Progress bar

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

Maybe not needed anymore if I just need to call `leaguedash` twice.

## Data cleaning

Originally did *some* data cleaning in the fetching stage. I'm beginning to think that all of it should be left alone, and any transformations should wait until the actual data cleaning and feature engineering stage.

When compiling, I thought of dropping some features which are by design collinear, but:

- I missed one, so I had to drop it in another data cleaning phase
- more feature engineering had to be done anyway, so data cleaning wasn't optional
- `fetch_league` should get the raw data and save the data cleaning afterwards

### `Transform`

The script should prepare the season's regular and playoffs stats so it can be fed directly to the model pipeline.

Scope:

- Remove players that do not meet the playtime threshold (to remove statistical noise)
- Feature engineer new variables
  - FG2M, FG2A
- Remove extraneous columns
  - \*\_PCT, FANATASY_PTS, FGM/A
- Merge regular and playoffs stat via a weighting coefficient and games played
  - Playoffs can be weighed at doubly as important as reg season
  - How about the ranks?
    - Concatenate as separate columns: `reg_rank` and `playoffs_rank`
    - Never mind; remove and re-rank after stat folding.
  - Minute played?
    - Heavily dependent on injuries and playoff success
    - Consider minutes per game
    - How to reconcile mpg between reg/playoffs? Same idea as the coefficients: Given a player with 20 mpg in 82 regular season games, and 10 mpg in 12 playoff games, the merged mpg would be calculated as
      $$\\text{mpg}\_{merge} = \\frac{20 * 82 + 2.0(10 * 12)}{82 + 2.0 * 12} = 18.1$$
  - If no playoff minutes?
    - Problem are the `*_RANK` fields. I dealt with this before by folding in the playoff stats into regular season - how do I fold ranks?
    - Re-do ranking after folding in stats? This loses the granularity that comes with separating regular season and playoff ranks
    - But I'm already doing that by folding the other numeric stats.

## Storage

Store locally for testing in `data/`, but eventually in a docker volume for portability, perhaps on S3
