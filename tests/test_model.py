import torch

from dep_car.model.lidar_dep import LidarDEPCarV1


def test_legacy_p0_lidar_prototype_remains_loadable_but_is_not_p4_architecture():
    model = LidarDEPCarV1()
    outputs = model(torch.zeros(2, 2, 16, 440), torch.zeros(2, 8))
    assert all(output.shape == (2, 15) for output in outputs)
    assert model.architecture_id == "dep_car_lidar_v1_3x5_mobilenetv3_v483"
    assert model.architecture_id != "dep_car_multimodal_v1_ackermann_3x5"
