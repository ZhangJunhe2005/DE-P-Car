import numpy as np

from dep_car.perception.bev import BEV_CHANNELS, LidarBEVConfig, build_lidar_bev
from dep_car.perception.pointcloud import SelfFilterConfig, filter_lidar_obstacles, filter_lidar_self_returns


def test_lidar_bev_keeps_front_side_and_rear_obstacles():
    config = LidarBEVConfig(extent=2.0, resolution=0.1)
    points = np.asarray([
        [1.0, 0.0, 0.2],
        [0.0, 1.0, 0.3],
        [-1.0, 0.0, 0.4],
        [0.0, -1.0, 0.5],
    ], dtype=np.float32)
    bev = build_lidar_bev(points, config)
    assert bev.shape == (len(BEV_CHANNELS), 40, 40)
    assert int(bev[0].sum()) == 4
    center = config.size // 2
    assert bev[5, center, center] == 1.0


def test_float32_point_just_inside_positive_extent_stays_in_last_cell():
    config = LidarBEVConfig(extent=8.0, resolution=0.1)
    boundary = np.nextafter(
        np.float32(config.extent), np.float32(0.0), dtype=np.float32
    )
    bev = build_lidar_bev(
        np.asarray([[boundary, boundary, 0.0]], dtype=np.float32), config
    )

    assert bev[0, -1, -1] == 1.0
    assert np.count_nonzero(bev[0]) == 1


def test_chassis_self_returns_are_removed_but_near_environment_is_retained():
    points = np.asarray([
        [-0.20, 0.05, 0.30],
        [0.20, -0.05, 0.20],
        [0.40, 0.00, 0.20],
        [0.00, 0.30, 0.20],
    ], dtype=np.float32)
    filtered = filter_lidar_self_returns(points, SelfFilterConfig())
    np.testing.assert_allclose(filtered, points[2:])


def test_obstacle_filter_rejects_ground_and_self_returns():
    points = np.asarray([
        [1.0, 0.0, -0.02],
        [-0.20, 0.0, 0.20],
        [1.0, 0.0, 0.20],
    ], dtype=np.float32)
    np.testing.assert_allclose(filter_lidar_obstacles(points), points[2:])
