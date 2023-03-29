import logging
import os
from typing import List

from pd.data_lab import PD_CLIENT_ORG_ENV, PD_CLIENT_STEP_API_KEY_ENV
from pd.management import Levelpak

from paralleldomain.data_lab import Location

logger = logging.getLogger(__name__)


def map_locations() -> List[Location]:
    locations = list()
    if all([n in os.environ for n in [PD_CLIENT_ORG_ENV, PD_CLIENT_STEP_API_KEY_ENV]]):
        for level in Levelpak.list():
            for version in level.versions:
                locations.append(Location(name=level.name, version=version))
    else:
        print("Missing step credentials! Will test with default map")
        locations.append(Location(name="Test_SF_6thAndMission_small_parking.umd"))
    return locations


LOCATIONS = {loc.name: loc for loc in map_locations()}
