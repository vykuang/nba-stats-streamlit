### Poetry

Kind of a pain on my desktop, I felt. Had issues getting poetry to recognize the version I set with pyenv (3.9.12), and not the system version (3.8.10). Especially strange since `poetry install` could find the correct pyenv version, and installed everything with that, but `poetry env info` then shows 3.8.10. Resolved by using `pyenv shell 3.9.12` prior to `poetry install`. The suggested fixes on poetry docs involved `poetry env use 3.9` to explicitly get poetry to recognize it, but no dice.
