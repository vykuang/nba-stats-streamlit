# Streamlit

The uber pickup tutorial gave a really good overview of what it can do: [Uber pickup tutorial](https://docs.streamlit.io/library/get-started/create-an-app)

It's got

- tabular view
- checkboxes to toggle which elements are visible
- charts
- sliders to filter those choarts
- maps if we give it lat/long

Pretty impressive stuff

## nba_stats

What do I want to show? Ultimately I'd like an interactive applet that lets the user chooose a player, run the pre-trained model on that player's stats, and return the label.

1. Raw data, after compilation?
1. Maybe a histogram for the three main stats - PTS, AST, REB, just to visualize the raw data a little
1. Compare the distributions of stats between the two seasons
1. Player selection
1. Predicted player label
1. Closest players from the comparison season
1. Visualization of how that player fits within that label?
   - Bar graph of each stat with that player and the chosen "closest" players from comparison season
   - How would this work?
     - AST vs REB, size=PTS scatterplot of all players with the label, with the selected player highlighted
   - Array of violinplots showing distribution of stats, indicating where that player was in relation to the current season vs comparison season
     - use slider for season
     - use toggles to select which stat to show?

## Execution

### Altair

Altair seems fairly intuitive to use. Similar to R's ggplot in some regards.

No easy way to violinplot? Need to use `.transform_density()` method which creates the histogram, and then re-orient it vertically and fill colors

Also, what can I do about the label names? Chaining the top three players as a singular label seems clunky, but hard to give context for those labels otherwise. Maybe I can limit the length of names chosen for these labels.

## Hurdles

Some challenges faced over the course of this project

### Data source

`cumeplayerstat` was my first choice for gathering player stats across an entire season, but couldn't figure out how to pass multiple game IDs, so defaulted back to a more basic endpoint in `playercareerstats`.

Next hurdle was the data extraction speed. Due to some unknown limit placed on the free public API, I had to reduce rate of request. Fetching the stats for every player for any given season would take tens of minutes, and doing so again for other seasons proved impractical if I wanted the app to respond in real time.

Not satisfied with the workflow, I dug around the endpoints and found `leaguedash`, which provides the stats across the whole season, for all players, with two additional perks:

1. One call returns every player
1. One call for the whole season

The ease of acquiring data from other seasons makes it feasible to start comparing players across eras

### Project layout

If I want to reuse code in my streamlit app, I run into the recurring issue of trying relative imports with no known parent package, since streamlit calls them as `__main__`. Need to rethink how to organize my code. Currently some repetitions between `train.py` and the `streamlit.py` app when it comes to `retrieve` and `reveal_group`. Perhaps I should keep `train.py` to do what the name suggests, and move out those latter two functions.

### Visualization

- Looking for an array of violin plots for season's distribution of stats
- Toggle between seasons
- Array of *bar graphs* displaying the selected player, similar players, and top/bottom players for each stat

## Docker
