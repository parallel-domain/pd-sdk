import logging
import os
from typing import List

from pd.management import Levelpak
from pd.data_lab.context import get_datalab_context
from paralleldomain.data_lab import Location

logger = logging.getLogger(__name__)


def map_locations() -> List[Location]:
    locations = list()
    context = get_datalab_context()
    if not context.is_mode_local:
        for level in Levelpak.list():
            for version in level.versions:
                locations.append(Location(name=level.name, version=version))
    else:
        print("Missing step credentials! Will test with default map")
        locations.append(Location(name="Test_SF_6thAndMission_small_parking.umd"))
    return locations


LOCATIONS = {loc.name: loc for loc in map_locations()}
