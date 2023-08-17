import sys

import numpy as np

from paralleldomain.model.geometry.bounding_box_3d import BoundingBox3DGeometry
from paralleldomain.model.occupancy import OccupancyGrid
from paralleldomain.utilities.coordinate_system import SIM_TO_INTERNAL
from paralleldomain.utilities.transformation import Transformation


def test_from_agent_bb3d():
    # Apply @ SIM_TO_INTERNAL.inverse to bring pose rotation from RFU to FLU so width/length/height are correct
    agent_1 = BoundingBox3DGeometry(
        pose=Transformation.from_euler_angles(translation=[0, 0, 0], angles=[0, 0, 0], order="xyz", degrees=True)
        @ SIM_TO_INTERNAL.inverse,
        width=1.0,
        height=1.0,
        length=1.0,
    )

    agent_2 = BoundingBox3DGeometry(
        pose=Transformation.from_euler_angles(translation=[10, 10, 0], angles=[0, 0, 0], order="xyz", degrees=True)
        @ SIM_TO_INTERNAL.inverse,
        width=5.0,
        height=1.0,
        length=10.0,
    )

    agent_3 = BoundingBox3DGeometry(
        pose=Transformation.from_euler_angles(translation=[0, 0, 50], angles=[0, 0, 0], order="xyz", degrees=True)
        @ SIM_TO_INTERNAL.inverse,
        width=1.0,
        height=1.0,
        length=1.0,
    )

    agent_4 = BoundingBox3DGeometry(
        pose=Transformation.from_euler_angles(translation=[100, 100, 10], angles=[0, 0, -45], order="xyz", degrees=True)
        @ SIM_TO_INTERNAL.inverse,
        width=0.5,
        height=1.0,
        length=20.0,
    )

    occupancy_grid = OccupancyGrid.from_bounding_boxes_3d(
        boxes=[
            agent_1,
            agent_2,
            agent_3,
            agent_4,
        ],
        resolution=0.1,
    )

    # POSITIVE TESTS

    # make sure that the center of each object is considered occupied
    assert np.all(
        occupancy_grid.is_occupied_world(
            points=np.asarray(
                [
                    agent_1.pose.translation[:2],
                    agent_2.pose.translation[:2],
                    agent_3.pose.translation[:2],
                    agent_4.pose.translation[:2],
                ]
            )
        )
    )

    # make sure half width is considered occupied
    assert np.all(
        occupancy_grid.is_occupied_world(
            points=np.asarray(
                [
                    [agent_1.pose.translation[0] + agent_1.width / 2, agent_1.pose.translation[1]],
                    [agent_2.pose.translation[0] + agent_2.width / 2, agent_2.pose.translation[1]],
                    [agent_3.pose.translation[0] + agent_3.width / 2, agent_3.pose.translation[1]],
                ]
            )
        )
    )

    # make sure negative half width is considered occupied
    assert np.all(
        occupancy_grid.is_occupied_world(
            points=np.asarray(
                [
                    [agent_1.pose.translation[0] - agent_1.width / 2, agent_1.pose.translation[1]],
                    [agent_2.pose.translation[0] - agent_2.width / 2, agent_2.pose.translation[1]],
                    [agent_3.pose.translation[0] - agent_3.width / 2, agent_3.pose.translation[1]],
                ]
            )
        )
    )

    # make sure half the length is considered occupied
    assert np.all(
        occupancy_grid.is_occupied_world(
            points=np.asarray(
                [
                    [agent_1.pose.translation[0], agent_1.pose.translation[1] + agent_1.length / 2],
                    [agent_2.pose.translation[0], agent_2.pose.translation[1] + agent_2.length / 2],
                    [agent_3.pose.translation[0], agent_3.pose.translation[1] + agent_3.length / 2],
                ]
            )
        )
    )

    # make sure negative half the length is considered occupied
    assert np.all(
        occupancy_grid.is_occupied_world(
            points=np.asarray(
                [
                    [agent_1.pose.translation[0], agent_1.pose.translation[1] - agent_1.length / 2],
                    [agent_2.pose.translation[0], agent_2.pose.translation[1] - agent_2.length / 2],
                    [agent_3.pose.translation[0], agent_3.pose.translation[1] - agent_3.length / 2],
                ]
            )
        )
    )

    # NEGATIVE tests

    # make sure out-of-bounds coordinates are considered unoccupied
    max_grid_y_index, max_grid_x_index = occupancy_grid._grid.shape
    (world_min_x, world_min_y), (world_max_x, world_max_y) = (
        occupancy_grid._grid_to_world(xy=np.asarray([[-1, -1]], dtype=int)).reshape(2).tolist(),
        occupancy_grid._grid_to_world(xy=np.asarray([[max_grid_x_index + 1, max_grid_y_index + 1]], dtype=int))
        .reshape(2)
        .tolist(),
    )

    assert np.all(
        np.logical_not(
            occupancy_grid.is_occupied_world(
                points=np.asarray(
                    [
                        [world_min_x - sys.float_info.epsilon, world_min_y - sys.float_info.epsilon],
                        [world_max_x, world_max_y],
                    ]
                )
            )
        )
    )

    # move on resolution step to the right (positive direction) and there shouldn't be occupancy
    assert np.all(
        np.logical_not(
            occupancy_grid.is_occupied_world(
                points=np.asarray(
                    [
                        [
                            agent_1.pose.translation[0] + agent_1.width / 2 + occupancy_grid._resolution,
                            agent_1.pose.translation[1],
                        ],
                        [
                            agent_2.pose.translation[0] + agent_2.width / 2 + occupancy_grid._resolution,
                            agent_2.pose.translation[1],
                        ],
                        [
                            agent_3.pose.translation[0] + agent_3.width / 2 + occupancy_grid._resolution,
                            agent_3.pose.translation[1],
                        ],
                    ]
                )
            )
        )
    )

    # move one resolution step to the left (negative direction) plus a small value, so we leave the cell
    # and there shouldn't be occupancy
    assert np.all(
        np.logical_not(
            occupancy_grid.is_occupied_world(
                points=np.asarray(
                    [
                        [
                            agent_1.pose.translation[0]
                            - agent_1.width / 2
                            - (occupancy_grid._resolution + sys.float_info.epsilon),
                            agent_1.pose.translation[1],
                        ],
                        [
                            agent_2.pose.translation[0]
                            - agent_2.width / 2
                            - (occupancy_grid._resolution + sys.float_info.epsilon),
                            agent_2.pose.translation[1],
                        ],
                        [
                            agent_3.pose.translation[0]
                            - agent_3.width / 2
                            - (occupancy_grid._resolution + sys.float_info.epsilon),
                            agent_3.pose.translation[1],
                        ],
                    ]
                )
            )
        )
    )

    # make sure half the length is considered occupied and there shouldn't be occupancy
    assert np.all(
        np.logical_not(
            occupancy_grid.is_occupied_world(
                points=np.asarray(
                    [
                        [
                            agent_1.pose.translation[0],
                            agent_1.pose.translation[1] + agent_1.length / 2 + occupancy_grid._resolution,
                        ],
                        [
                            agent_2.pose.translation[0],
                            agent_2.pose.translation[1] + agent_2.length / 2 + occupancy_grid._resolution,
                        ],
                        [
                            agent_3.pose.translation[0],
                            agent_3.pose.translation[1] + agent_3.length / 2 + occupancy_grid._resolution,
                        ],
                    ]
                )
            )
        )
    )

    # move one resolution step to the left (negative direction) plus a small value, so we leave the cell
    # and there shouldn't be occupancy
    assert np.all(
        np.logical_not(
            occupancy_grid.is_occupied_world(
                points=np.asarray(
                    [
                        [
                            agent_1.pose.translation[0],
                            agent_1.pose.translation[1]
                            - agent_1.length / 2
                            - (occupancy_grid._resolution + sys.float_info.epsilon),
                        ],
                        [
                            agent_2.pose.translation[0],
                            agent_2.pose.translation[1]
                            - agent_2.length / 2
                            - (occupancy_grid._resolution + sys.float_info.epsilon),
                        ],
                        [
                            agent_3.pose.translation[0],
                            agent_3.pose.translation[1]
                            - agent_3.length / 2
                            - (occupancy_grid._resolution + sys.float_info.epsilon),
                        ],
                    ]
                )
            )
        )
    )


def test_grid_world_conversion():
    # Test different grid resolutions and check that world->grid->world resolves within resolution limit
    for grid_resolution in [0.01, 0.1, 0.5, 1.0, 10.0]:
        occupancy_grid_0_10 = OccupancyGrid(
            width=100.0, height=100.0, offset_x=50.0, offset_y=50.0, resolution=grid_resolution
        )

        world_in = np.asarray(
            [
                [100, 100],
                [243.2, 520.2],
                [0.0, 0.0],
            ]
        )
        grid_out = occupancy_grid_0_10._world_to_grid(xy=world_in)
        world_out = occupancy_grid_0_10._grid_to_world(xy=grid_out)

        world_equal = np.isclose(world_in, world_out, atol=grid_resolution)
        assert np.all(world_equal)

    # Use a finer tolerance than grid resolution and see that none of the values are converted back correctly
    # by using 1/10 of resolution as tolerance, and having that as decimal world coordinates, we should see
    # too much truncation in conversion which results in `np.isclose` being `False`

    grid_resolution = 0.1
    occupancy_grid_0_10 = OccupancyGrid(
        width=100.0, height=100.0, offset_x=50.0, offset_y=50.0, resolution=grid_resolution
    )

    world_in = np.asarray(
        [
            [100.35, 100.45],
            [243.23, 520.23],
            [0.57, -0.77],
        ]
    )
    grid_out = occupancy_grid_0_10._world_to_grid(xy=world_in)
    world_out = occupancy_grid_0_10._grid_to_world(xy=grid_out)

    world_equal = np.isclose(world_in, world_out, atol=grid_resolution / 10)
    assert np.all(np.logical_not(world_equal))
