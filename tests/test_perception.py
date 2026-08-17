import numpy as np

from dep_car.perception.range_image import build_range_image


def test_range_image_contract_and_nearest_return():
    points = np.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    image, mask = build_range_image(points, azimuth_bins=440, max_range=40.0)
    assert image.shape == mask.shape == (16, 440)
    assert mask.sum() == 1.0
    assert np.isclose(image[mask.astype(bool)][0], 1.0 / 40.0)

