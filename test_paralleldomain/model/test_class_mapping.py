import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap, LabelMapping, OnLabelNotDefined


class TestClassMap:
    def test_id_to_str(self):
        class_map = ClassMap.from_id_label_dict({i: str(i + 2) for i in range(42)})

        for i in range(42):
            assert class_map[i].name == str(i + 2)

        with pytest.raises(KeyError):
            _ = class_map[42]


class TestLabelMapping:
    def test_label_mapping(self):
        label_mapping = {"Car": "car", "Pedestrian": "ped", "Rider": "ped"}
        label_map = LabelMapping(label_mapping=label_mapping, on_not_defined=OnLabelNotDefined.RAISE_ERROR)
        for label in list(label_mapping.keys()):
            assert label_map[label] == label_mapping[label]

        with pytest.raises(KeyError):
            _ = label_map["Bus"]

    def test_label_mapping_return_identity(self):
        label_mapping = {"Car": "car", "Pedestrian": "ped", "Rider": "ped"}
        label_map = LabelMapping(label_mapping=label_mapping, on_not_defined=OnLabelNotDefined.KEEP_LABEL)
        for label in list(label_mapping.keys()):
            assert label_map[label] == label_mapping[label]

        for label in ["a", "b", "c"]:
            assert label_map[label] == label

    def test_label_mapping_discard(self):
        label_mapping = {"Car": "car", "Pedestrian": "ped", "Rider": "ped"}
        label_map = LabelMapping(label_mapping=label_mapping, on_not_defined=OnLabelNotDefined.DISCARD_LABEL)
        for label in list(label_mapping.keys()):
            assert label_map[label] == label_mapping[label]

        for label in ["a", "b", "c"]:
            assert label_map[label] is None

    def test_map_chaining(self):
        label_mapping = {"Car": "car", "Pedestrian": "ped", "Rider": "rider"}
        label_map = LabelMapping(label_mapping=label_mapping, on_not_defined=OnLabelNotDefined.RAISE_ERROR)
        label_mapping2 = {"car": "thing", "ped": "thing", "rider": "Rider"}
        label_map2 = LabelMapping(label_mapping=label_mapping2, on_not_defined=OnLabelNotDefined.RAISE_ERROR)

        chained_map = label_map2 @ label_map
        for label in list(label_mapping.keys()):
            if label == "Rider":
                assert chained_map[label] == "Rider"
            else:
                assert chained_map[label] == "thing"


class TestClassIdMap:
    def test_map_ids(self):
        custom_id_map = ClassIdMap(class_id_to_class_id={i: i + 2 for i in range(100)})

        for i in range(100):
            assert custom_id_map[i] == i + 2

    def test_map_key_error(self):
        custom_id_map = ClassIdMap(class_id_to_class_id={i: i + 2 for i in range(100)})

        with pytest.raises(KeyError):
            _ = custom_id_map[2222]

    def test_map_numpy_array(self):
        custom_id_map = ClassIdMap(class_id_to_class_id={i: i + 2 for i in range(100)})
        source = np.arange(0, 5)
        target = np.arange(2, 7)
        mapped = custom_id_map[source]
        assert np.all(target == mapped)

        source = np.arange(0, 9).reshape((3, 3, 1))
        target = np.arange(2, 11).reshape((3, 3, 1))
        mapped = custom_id_map[source]
        assert np.all(target == mapped)
