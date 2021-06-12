import pytest
from paralleldomain.model.class_mapping import ClassMap, LabelMapping, OnLabelNotDefined


class TestClassMap:
    def test_id_to_str(self):
        class_map = ClassMap(class_id_to_class_name={i: str(i + 2) for i in range(42)})

        for i in range(42):
            assert class_map[i] == str(i + 2)

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
            assert label_map[label] == None

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
