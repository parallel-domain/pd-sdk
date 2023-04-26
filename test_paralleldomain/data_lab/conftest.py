import os

import pytest

from pd.data_lab.context import setup_datalab


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    setup_datalab("v2.0.0-beta")
