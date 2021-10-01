import numpy as np

from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.model.class_distribution import ClassDistribution


def test_from_dataset(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()
    class_dist = ClassDistribution.from_dataset(dataset=dataset)
    assert class_dist is not None
    assert isinstance(class_dist, ClassDistribution)
    car_info = class_dist.get_class_info(class_name="Car")
    sum = 0
    for ci in class_dist.class_distribution_infos:
        sum += ci.class_pixel_percentage
    assert np.allclose(sum, 100.0)
    assert car_info is not None
    assert car_info.class_pixel_count > 0
    assert car_info.class_pixel_percentage > 1.0
    assert car_info.class_instance_count > 0
    assert car_info.class_instance_percentage > 1.0
