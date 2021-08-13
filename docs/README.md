# How to extend and build PD SDK API Docs?

## Setup _sphinx_ and dependencies

In your virtual environment, `cd` into this directory (`docs/`) and run

`pip install -r requirements.txt`

## Build documentation

In your virtual environment, `cd` into this directory (`docs/`) and run

`sphinx-build -b html source build`

The output will be written into `docs/build`.

## Extend documentation

Follow the Google style docstrings, as documented [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Once you have documented a new file, make sure you add it in `docs/source/index.rst` using the `.. automodule` command.
