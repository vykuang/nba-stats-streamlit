# nba-stats-streamlit

A basic end-to-end deployment of a simple model using data from NBA stats.

## Brainstorming

- Given a player's season average, classify the type of player.
  - Train on the 2021-2022 season's averages for each player
  - Clustering to reveal the groups, if any
  - User PUTs their request in the form of a collection of seasonal stat averages, and the model should respond with the predicted class of that player, along with some comparisons from the training set
- Seems cumbersome and altogether not really worthwhile to enter some stats of a fictitious player to see a comparison
  - User selects the following:
    1. season, e.g. `2020-21`
    1. player from that season in a dropdown menu
    1. another season from which the user wants to draw player comparison
  - Model returns the top three most similar players from the comparison season
  - For example if I select:
    1. `2004-05`
    1. `Kobe Bryant`
    1. `1995-96`
       I might expect to see `Michael Jordan`, but not `David Robinson`

## Architecture

- sklearn trains our model
- mlflow tracks experiments
- Flask acts as basic backend (streamlit will request for the different resources via this backend)
- Streamlit as frontend
- poetry manages dependencies
- docker containerizes each service

## What will streamlit do?

Streamlit will act is the frontend interface that the end-user interacts with.

```
* User will select the input parameters:
    * player,
    * player season,
    * comparison season
* Streamlit will have a visualization template for the season stats, and how they might differ or overlap with each other
    * Distributions of key stats across the seasons, e.g. PTS, 3GA via violion plots
    * Run the pre-trained model to find similar players based on user input
    * Visualize and contrast the player comps on a scatterplot
        * Include the label's top players in the comparison scatterplot
    * Bar graph to view comparisons of individual stat of:
        * selected player
        * similar player
        * lowest and top ranked player from both current and comparison season?
* [optional] Add toggle to select comparison stat. Perhaps we only want PFD (personal fouls drawn), FTM, FTA
```

## How do we get there?

Streamlit will require:

- Trained model, perhaps in `pickle` form
- `leaguedash` data for all the seasons
  - Do we pre-download everything first? Have that be part of the initialization?
  - Doesn't make much sense to request it ad-hoc for every request
  - Perhaps we *could* do that, if we also save those results so it can be reused for later queries

### Deciding player similarity

The stat ranks will play a large part in determining the similarity, but only after the label is revealed by the model. After that we could feature-engineer some kind of aggregate ranking to be applied intra-label, and then pick players from the comparison season adjacent to the selected player's rank in the current season

`PLUS_MINUS_RANK` and `MIN_RANK` to be used for aggregate ranking, emphasizing player impact and in-game time.
