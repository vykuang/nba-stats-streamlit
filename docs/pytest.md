# Pytest

## Note on `src` layout

Been over this a million times but still worth it to recap.

[Per pytest good integration practices](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html), we're expected to test against the *installed* version of our package, rather than the local version. To do so, do this:

```bash
cd <proj_root>
pip install --editable .
```

Which allows us to edit our package, and rerun tests against those edits. Tucking our code inside `/src` insulates pytest from testing local code, and forces it to test against the installed version

I don't think I will follow it for this project since it's not intended to be distributed as a package.

## Unittests

What unit tests are we running here?

* data fetch
    * test json response; requires online connection and uses up some arbitrary read request limit
* data preprocessing
    * prepare some default json response for transformations
    * implement mocking/monkeypatching where necessary
* orchestration
    * test that the wrapped prefect function works as intended?
    *

## Monkeypatch vs mock

Monkey patching refers to replacing a function/method/class by another at runtime. Purpose could be for testing, bugfix, or generally changing the behaviour.

In a very similar vein, **mocking** achieves the same objective, and ultimately comes down to preference which one is used

Generally both can be used so that third party calls aren't relied upon whenever we want to run tests.

Other features:

* set/del class attributes
* set/del dict
* set/del environment vars.

### mocking API response with `monkeypatch.setattr`

Example func that calls API:

```py
# app.py
import requests

def get_json(url):
    """Takes URL and returns json"""
    r = requests.get(url)
    return r.json()
```

When we test, we don't want to actually make the API call, so we *monkeypatch* the `requests.get()` to return a ready-made response instead:

```py
# test_app.py
# for us to monkeypatch
import requests

# the code we're testing
import app

# custom class that returns the mock json, i.e. dict
# when the monkeypatched requests.get() returns a response
class MockResponse:
    # our mock json() always returns a hard-coded dict
    @staticmethod
    def json():
        return {"mock_key": "mock_response"}

def test_get_json(monkeypatch):
    # regardless of whatever arg is passed to get_json(),
    # it will always return our mock response because
    # of the monkeypatch
    def mock_get(*args, **kwargs):
        return MockResponse()

    # monkeypatching the requests method and changing
    # to our mock_get
    monkeypatch.setattr(requests, "get", mock_get)

    # now when we call get_json, which in turn calls
    # requests.get(), it will instead call mock_get
    # and turns instance of our MockResponse object, which we defined
    # above this unit test
    # No API requests made.
    result = app.get_json("https://some_url.abc")
    assert result["mock_key"] == "mock_response"
```

Not sure about the `@staticmethod` thing in our `MockResponse` class definition.

How would we define it in our `nba_api` tests?

```py
# fetch.py

from nba_api.stats.endpoints import playercareerstats

def get_player_season(player_id):
    """Retrieves player's season averages from nba's API"""
    # this is where the call happens
    seasons = playercareerstats.PlayerCareerStats(player_id)
    return seasons.get_normalized_json()

```

So now we have to monkeypatch the method inside `playercareerstats`? When that class is instantiated, API call is automatically made. We want to monkeypatch that specific call so that it doesn't happen, and instead calls our monkeypatch function which returns a pre-made json response. The mock class we make, `MockResponse` should have a corresponding `.get_normalized_json()` static method with the hard-coded dict.

```py
# test_get_player_season.py
from nba_api... import

def test_get_season(monkeypatch):
    def mock_get_career(*args, **kwargs):
        """Replaces .get_request(), so that our test does not
        make API call to stats.nba.com
        This func will just not do anything.
        It takes in any amount of args, but returns nothing
        since original arg also returns nothing, but changes
        the class's self.properties.
        """

    def mock_get_json(*args, **kwargs):
        """When we call career_stats.get_normalized_json(), it will
        call this func instead
        """
        return {"mock_header": ["mock1", "mock2"]}

    # patching the .get_request() instance method
    monkeypatch.setattr(
        playercareerstats.PlayerCareerStats,
        "get_response",
        mock_get_career,
    )
    # patching the .get_normalized_json() instance method
    monkeypatch.setattr(
        playercareerstats.PlayerCareerStats,
        "get_normalized_json",
        mock_get_json,
    )
    test_id = 2544
    response = fetch.get_career_stats(player_id=test_id)
    assert response["mock_header"] == ["mock1", "mock2"]
```

## Logging

Pytest automatically displays `WARNING` level and above logs to console.

Use `--log-cli-level=DEBUG` or any other level to explicitly set the severity for that particular pytest run
