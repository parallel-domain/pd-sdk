import os

import pd.management
import pytest
from pd.data_lab.constants import PD_CLIENT_ORG_ENV, PD_CLIENT_STEP_API_KEY_ENV


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    if all([n in os.environ for n in [PD_CLIENT_ORG_ENV, PD_CLIENT_STEP_API_KEY_ENV]]):
        pd.management.org = os.environ[PD_CLIENT_ORG_ENV]
        pd.management.api_key = os.environ[PD_CLIENT_STEP_API_KEY_ENV]
