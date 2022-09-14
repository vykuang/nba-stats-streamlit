# Data Ingestion

Data is sourced from NBA's stats API. To facilitate this, `nba_api` will be imported so python can programmatically fetch and store the requested json data inside a postgres container

## Endpoints

NBA's stats API offer a lot of endpoints. Since we're interested in a player's season averages, we are looking at `cumestatsplayer` endpoint, and restricting it to all games within 21-22 season.
