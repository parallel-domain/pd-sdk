import numpy as np
import pytest

from paralleldomain.model.class_mapping import ClassIdMap, ClassMap, ClassNameToIdMap, LabelMapping, OnLabelNotDefined


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


class TestEndToEndMapping:
    def test_np_class_ids_mapping(self):
        label_mapping = {"Car": "vehicle", "Pedestrian": "ped", "Pickup": "vehicle", "Truck": "vehicle"}
        label_map = LabelMapping(label_mapping=label_mapping, on_not_defined=OnLabelNotDefined.RAISE_ERROR)
        class_map = ClassMap.from_id_label_dict({0: "Car", 1: "Pickup", 2: "Truck", 3: "Pedestrian"})
        name_to_id_map = ClassNameToIdMap(name_to_class_id={"vehicle": 1, "ped": 0})
        class_ids = np.array([1, 2, 3, 0, 1, 0, 3, 2, 3])
        target_class_ids = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0])

        mapped_labels = label_map @ class_map
        assert isinstance(mapped_labels, ClassMap)

        class_mapping = name_to_id_map @ mapped_labels
        assert isinstance(class_mapping, ClassIdMap)

        mapped_class_ids = class_mapping @ class_ids
        assert isinstance(mapped_class_ids, np.ndarray)
        assert np.all(mapped_class_ids == target_class_ids)

        mapped_class_ids = class_mapping @ class_ids.reshape((3, 3))
        assert isinstance(mapped_class_ids, np.ndarray)
        assert np.all(mapped_class_ids.flatten() == target_class_ids)
