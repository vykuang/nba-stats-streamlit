# Pytest

Developing with testing as a priority is ESSENTIAL so that whenever you make changes to the code, there's a surefire way of knowing the changes don't break the code.

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

- data fetch
  - test json response; requires online connection and uses up some arbitrary read request limit
- data preprocessing
  - prepare some default json response for transformations
  - implement mocking/monkeypatching where necessary
- orchestration
  - test that the wrapped prefect function works as intended?
  -

## Monkeypatch vs mock

Monkey patching refers to replacing a function/method/class by another at runtime. Purpose could be for testing, bugfix, or generally changing the behaviour.

In a very similar vein, **mocking** achieves the same objective, and ultimately comes down to preference which one is used

Generally both can be used so that third party calls aren't relied upon whenever we want to run tests.

Other features:

- set/del class attributes
- set/del dict
- set/del environment vars.

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

### Log to console

To log to console, add a `streamHandler` for `sys.stdout`:

```py
# try_log.py
import logging
import sys

def foo(num: int = 5, log_level: int = logging.DEBUG):
    logger = logging.getLogger('foo')
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    logger.addHandler(handler)

    for i in range(num):
        logger.debug(f'debug: {i}')
        logger.info(f'info: {i}')
```

### Common practice

In each module, instantiate a logger using `__name__`, and add a handler.

```py
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
```

Then parametrize the level of logging via env var or CLI options in the wrapper script

```py
def app(loglevel: str = 'info'):
    num_loglevel = getattr(logging, loglevel, None)

    if not isinstance(num_loglevel, int):
        raise ValueError(f"Invalid log level: {loglevel}")

    logger.setLevel(num_loglevel)
    handler.setLevel(num_loglevel)

    logger.addHandler(handler)

    # do script
```

#### Format the logger output

In pytest.ini, or in pyproject.toml, customize logger output.

```ini
[tool.pytest.ini_options] # for py.toml
[pytest] # for ini
log_format = "%(asctime)s %(levelname)s %(message)s"
log_date_format = "%Y-%m-%d %H:%M:%S"
```

## Validating data pipeline

In order to validate the pipeline we need to make sure the transformations are doing what they're supposed to. One way is to have a mock dataset, and a mock result to confirm the function output matches the expected mock output. That means we need to generate the expected output for each step in the transformation.

Use a notebook to do so interactively???

## Fixtures

Through using fixtures, and wanting to return the two versions of file paths for regular and post-season stat pickles/df, I came across *Factories as fixtures* design pattern that can parametrize my path fixture, without repeating code. [From docs](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#factories-as-fixtures):

```py
@pytest.fixture
def make_customer_record():

    created_records = []

    def _make_customer_record(name):
        record = models.Customer(name=name, orders=[])
        created_records.append(record)
        return record

    yield _make_customer_record

    for record in created_records:
        record.destroy()


def test_customer_records(make_customer_record):
    customer_1 = make_customer_record("Lisa")
    customer_2 = make_customer_record("Mike")
    customer_3 = make_customer_record("Meredith")

## my implementation
@pytest.fixture
def make_league_pickle():
    def _make_league_pickle(season_type: str = "regular"):
        return test_data_dir / f"leaguedash_{season_type}_2018-19.pkl"
    return _make_league_pickle
```

This is different from actually *parametrizing* the fixtures, which repeats the tests that request those fixtures by return each variant of the fixture. [Parametrizing fixtures](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#parametrizing-fixtures)

Not what I need right now, since I need both regular and post-season for the merge tests. However for others, parametrizing would work - the tests would simply run on both regular season and post.

## pytest-flask

The plugin `pytest-flask` provides useful fixtures in the form of flask contexts within which to test our flask application.

[pytest-flask docs here](https://pytest-flask.readthedocs.io/en/latest/features.html)

But flask itself already provides good documentation and tools for testing: [docs here](https://flask.palletsprojects.com/en/2.2.x/tutorial/tests/)

Specifically, `app.test_client()` and `app.test_cli_runner` are essential for testing in flask.

However I would want to integrate docker in the flask functional tests too, since ultimately the backend will run in the containers, not local env

## pytest-docker

Uses docker-compose to spin up container fixtures

## pytest-mlflow

This plugin doesn't actually exist, and preliminary searches for testing re: mlflow has been bare.

Idea: I can still test the underlying test functions if I set up a local `mlflow.db` and `./mlruns` folder as the tracking URI and artifact root. I may not be testing against a remote tracking server, but this way allows me to continue unit testing as normal.

Re-use the `test_gametime_df.pkl` as the test input for my model unit tests. `/model` takes these args:

- `season`
- `data_path`
- `loglevel`
- `max_evals`

In addition, set these environment vars via `monkeypatch.setenv`:

- `MLFLOW_TRACKING_URI=tmp_path / mlflow.db`
- `MLFLOW_EXP_NAME=test`
- `MLFLOW_REGISTERED_MODEL=test-clusterer`
- `MLFLOW_ARTIFACT_PATH=test_model`

Conversely, test absence of or incorrect env vars with `monkeypatch.delenv`

Initial tests did not successfully set the test environment vars

In my `train` function I've already called the env var and set it to the above constants. Setting new env vars after the fact does not change their value, and it did not affect the original assignment; they only took from the defaults because those are determined at *import time*, and I patched the env at *runtime*.

I can try patching the variables, but `mlflow.set...()` would also happen at import time

But I need the top global assignment to use my `mlflow.MlflowClient` freely throughout the module.

Do I need to refactor the code to remove the global scoped variables?

If I can replace `mlflow.start_run()` and other *high level fluent API* I could perhaps pass the client around, and use it as fixture, but the client involves lower level API that just makes everything more convoluted. Most notably it only has `.log_param(run_id, param_name, param_val)`, not `mlflow.log_params(params_dict)`.

General tips from [this succinct SOF post](https://stackoverflow.com/a/73579245/5496416)

> - don't execute `os.getenv()` at import time (i.e. not global)
>
> - monkeypatch the `VAR` instead of `"ENV_VAR"`
>
> - mock env in `pytest_sessionstart`??? Not easily showing up in searches.
>
> - `pytest` executes in the following order:
>
>   1. import `conftest`
>   1. executes that
>   1. import tests
>   1. execute tests

Those env vars are set at import time whereas our monkeypatch fixture is executed after loading the test modules.

[From this discussion chain](https://github.com/pytest-dev/pytest/issues/2229) there seems to be some irreconcilable differences between the pattern of *global variable as central configs* vs pytest philosophy.

A kind of a braindead way is to just run this globally in `conftest`:

```py
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///tests/data/mlflow.db"
```

But obviously now we don't have access to `tmp_path`

Perhaps refactor my code to avoid those global configs.

- set `mlflow.set_...` inside `model()`
- set `MlflowClient(...)` inside `find` and `register`
- in my pytest.fixture, use `monkeypatch.setattr(model, "VAR_NAME", value)`, where `model` is from `from frontend.model import model`
