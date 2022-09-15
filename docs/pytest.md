# Pytest

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
