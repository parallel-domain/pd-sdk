import os

import pytest

from test_paralleldomain.data_lab.constants import LOCATION_VERSION
from pd.data_lab.context import setup_datalab


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    setup_datalab(LOCATION_VERSION)
