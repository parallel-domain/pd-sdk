import logging

import numpy as np
import pytest
from pd.umd import load_umd_map

from paralleldomain.data_lab.config.map import MapQuery, UniversalMap
from paralleldomain.utilities.transformation import Transformation
from test_paralleldomain.data_lab.constants import LOCATIONS

logger = logging.getLogger(__name__)


class TestUMDMap:
    @pytest.mark.parametrize("location_name", list(LOCATIONS.keys()))
    def test_can_draw_random_pose_deterministically(self, location_name: str):
        location = LOCATIONS[location_name]
        map_file = load_umd_map(name=location.name, version=location.version)
        map = UniversalMap(proto=map_file)

        query = MapQuery(map=map)

        random_pose_1 = query.get_random_street_location(random_seed=23)

        assert random_pose_1 is not None
        assert isinstance(random_pose_1, Transformation)

        random_pose_2 = query.get_random_street_location(random_seed=23)
        assert random_pose_2 is not None
        assert isinstance(random_pose_2, Transformation)
        # Test if the same pose is returned. If so the inverse should be the same as the identity
        np.allclose(np.eye(4), (random_pose_2.inverse @ random_pose_1).transformation_matrix)
