import pytest

from paralleldomain.model.annotation import AnnotationTypes, AnnotationIdentifier, AnnotationType


class TestAnnotationIdentifier:
    @pytest.mark.parametrize(
        "annotation_type",
        [
            AnnotationTypes.OpticalFlow,
            AnnotationTypes.BackwardOpticalFlow,
            AnnotationTypes.InstanceSegmentation2D,
            AnnotationTypes.SemanticSegmentation2D,
            AnnotationTypes.SurfaceNormals3D,
            AnnotationTypes.Albedo2D,
            AnnotationTypes.BackwardSceneFlow,
            AnnotationTypes.BoundingBoxes2D,
            AnnotationTypes.BoundingBoxes3D,
            AnnotationTypes.Depth,
            AnnotationTypes.InstanceSegmentation3D,
            AnnotationTypes.MaterialProperties2D,
            AnnotationTypes.MaterialProperties3D,
            AnnotationTypes.PointCaches,
            AnnotationTypes.SemanticSegmentation3D,
        ],
    )
    def test_hashing(self, annotation_type: AnnotationType):
        test_dict = {annotation_type: 123}
        default_identifier = AnnotationIdentifier(annotation_type=annotation_type)
        assert hash(annotation_type) == hash(default_identifier)
        assert default_identifier in test_dict
        assert test_dict[default_identifier] == 123
        assert default_identifier in [annotation_type]

        named_identifier = AnnotationIdentifier(annotation_type=annotation_type, name="random_name")
        assert hash(annotation_type) != hash(named_identifier)
        assert named_identifier not in test_dict
        assert named_identifier not in [annotation_type]

        named_identifier2 = AnnotationIdentifier(annotation_type=annotation_type, name="random_name")
        test_dict[named_identifier] = 124
        assert hash(named_identifier2) == hash(named_identifier)
        assert named_identifier2 in test_dict
        assert test_dict[named_identifier2] == 124
        assert named_identifier2 in [named_identifier]

        other_named_identifier = AnnotationIdentifier(annotation_type=annotation_type, name="!random_name")
        assert hash(named_identifier) != hash(other_named_identifier)
        assert other_named_identifier not in test_dict
        assert other_named_identifier not in [annotation_type, named_identifier, named_identifier2]

    def test_resolve_annotation_identifier(self):
        with pytest.raises(ValueError):
            AnnotationIdentifier.resolve_annotation_identifier(
                available_annotation_identifiers=[], annotation_type=AnnotationTypes.BoundingBoxes2D
            )

        with pytest.raises(ValueError):
            AnnotationIdentifier.resolve_annotation_identifier(
                available_annotation_identifiers=[],
                annotation_identifier=AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D),
            )

        resolved = AnnotationIdentifier.resolve_annotation_identifier(
            available_annotation_identifiers=[AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D)],
            annotation_identifier=AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D),
        )
        assert resolved.annotation_type == AnnotationTypes.BoundingBoxes2D
        assert resolved.name is None

        resolved = AnnotationIdentifier.resolve_annotation_identifier(
            available_annotation_identifiers=[
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="testName"),
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D, name="otherName"),
            ],
            annotation_identifier=AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D),
        )
        assert resolved.annotation_type == AnnotationTypes.BoundingBoxes2D
        assert resolved.name == "testName"

        resolved = AnnotationIdentifier.resolve_annotation_identifier(
            available_annotation_identifiers=[
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="testName"),
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D, name="otherName"),
            ],
            annotation_type=AnnotationTypes.BoundingBoxes2D,
        )
        assert resolved.annotation_type == AnnotationTypes.BoundingBoxes2D
        assert resolved.name == "testName"

        resolved = AnnotationIdentifier.resolve_annotation_identifier(
            available_annotation_identifiers=[
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="testName"),
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D, name="otherName"),
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="anotherName"),
            ],
            annotation_type=AnnotationTypes.BoundingBoxes2D,
            name="testName",
        )
        assert resolved.annotation_type == AnnotationTypes.BoundingBoxes2D
        assert resolved.name == "testName"

        resolved = AnnotationIdentifier.resolve_annotation_identifier(
            available_annotation_identifiers=[
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="testName"),
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D, name="otherName"),
                AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="anotherName"),
            ],
            annotation_identifier=AnnotationIdentifier(
                annotation_type=AnnotationTypes.BoundingBoxes2D,
                name="testName",
            ),
        )
        assert resolved.annotation_type == AnnotationTypes.BoundingBoxes2D
        assert resolved.name == "testName"

        with pytest.raises(ValueError):
            AnnotationIdentifier.resolve_annotation_identifier(
                available_annotation_identifiers=[
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="testName")
                ],
                annotation_type=AnnotationTypes.BoundingBoxes2D,
                name="otherName",
            )

        with pytest.raises(ValueError):
            AnnotationIdentifier.resolve_annotation_identifier(
                available_annotation_identifiers=[
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="testName"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="otherName"),
                ],
                annotation_type=AnnotationTypes.BoundingBoxes2D,
            )
