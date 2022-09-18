# Data Ingestion

Data is sourced from NBA's stats API. To facilitate this, `nba_api` will be imported so python can programmatically fetch and store the requested json data inside a postgres container

## Endpoints

NBA's stats API offer a lot of endpoints. Since we're interested in a player's season averages, we are looking at `cumestatsplayer` endpoint, and restricting it to all games within 21-22 season.

Couldn't get it to work; unable to pass multiple game_IDs to fetch the stats. The composed request URL contains all the IDs, but only the first one is accepted.

`playercareerstats` used instead.

To gather the player_ids, use the `static` endpoint to gather all active players.

## Pipeline

1. Gather all active player_ids via static endpoint
2. For each player_id, request `playercareerstats` API.
3. Check that `MIN` played > 500 minutes, or 'GP' > 40; remove others
4. Fold the SeasonTotalsRegular and SeasonTotalsPost to a single dict
5. Store all player's data into .json for later storage into postgres
