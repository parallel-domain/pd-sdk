import logging
from typing import List

from pd.data_lab.context import get_datalab_context, setup_datalab
from pd.management import Levelpak

from paralleldomain.data_lab import Location
from paralleldomain.data_lab import DEFAULT_DATA_LAB_VERSION

logger = logging.getLogger(__name__)

LOCATION_VERSION = DEFAULT_DATA_LAB_VERSION
setup_datalab(LOCATION_VERSION)


def map_locations() -> List[Location]:
    locations = list()
    context = get_datalab_context()
    if not context.is_mode_local:
        for level in Levelpak.list():
            for version in level.versions:
                # Skipping this map since it is broken
                if level.name == "Test_SF_6thAndMission_small":
                    continue
                if version == LOCATION_VERSION:
                    locations.append(Location(name=level.name, version=version))
    else:
        print("Missing step credentials! Will test with default map")
        locations.append(Location(name="Test_SF_6thAndMission_small_parking.umd"))
    return locations


LOCATIONS = {loc.name: loc for loc in map_locations()}
