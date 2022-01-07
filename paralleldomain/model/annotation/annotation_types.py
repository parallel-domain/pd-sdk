from typing import Type

from paralleldomain.model.annotation.albedo_2d import Albedo2D
from paralleldomain.model.annotation.bounding_box_2d import BoundingBoxes2D
from paralleldomain.model.annotation.bounding_box_3d import BoundingBoxes3D
from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.annotation.depth import Depth
from paralleldomain.model.annotation.instance_segmentation_2d import InstanceSegmentation2D
from paralleldomain.model.annotation.instance_segmentation_3d import InstanceSegmentation3D
from paralleldomain.model.annotation.material_properties_2d import MaterialProperties2D
from paralleldomain.model.annotation.optical_flow import OpticalFlow
from paralleldomain.model.annotation.point_2d import Points2D
from paralleldomain.model.annotation.point_cache import PointCaches
from paralleldomain.model.annotation.polygon_2d import Polygons2D
from paralleldomain.model.annotation.polyline_2d import Polylines2D
from paralleldomain.model.annotation.scene_flow import SceneFlow
from paralleldomain.model.annotation.semantic_segmentation_2d import SemanticSegmentation2D
from paralleldomain.model.annotation.semantic_segmentation_3d import SemanticSegmentation3D
from paralleldomain.model.annotation.surface_normals_2d import SurfaceNormals2D
from paralleldomain.model.annotation.surface_normals_3d import SurfaceNormals3D

AnnotationType = Type[Annotation]


class AnnotationTypes:
    """Allows to get type-safe access to annotation type related information, e.g., annotation data or class maps.

    Attributes:
        BoundingBoxes2D
        BoundingBoxes3D
        SemanticSegmentation2D
        InstanceSegmentation2D
        SemanticSegmentation3D
        InstanceSegmentation3D
        OpticalFlow
        Depth
        SurfaceNormals3D
        SurfaceNormals2D
        SceneFlow
        MaterialProperties2D
        Albedo2D
        Points2D
        Polygons2D
        Polylines2D

    Examples:
        Access 2D Bounding Box annotations for a camera frame:
        ::

            camera_frame: SensorFrame = ...  # get any camera's SensorFrame

            from paralleldomain.model.annotation import AnnotationTypes

            boxes_2d = camera_frame.get_annotations(AnnotationTypes.BoundingBoxes2D)
            for b in boxes_2d.boxes:
                print(b.class_id, b.instance_id)

        Access class map for an annotation type in a scene:
        ::

            scene: Scene = ...  # get a Scene instance

            from paralleldomain.model.annotation import AnnotationTypes

            class_map = scene.get_class_map(AnnotationTypes.SemanticSegmentation2D)
            for id, class_detail in class_map.items():
                print(id, class_detail.name)
    """

    BoundingBoxes2D: Type[BoundingBoxes2D] = BoundingBoxes2D  # noqa: F811
    BoundingBoxes3D: Type[BoundingBoxes3D] = BoundingBoxes3D  # noqa: F811
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D  # noqa: F811
    InstanceSegmentation2D: Type[InstanceSegmentation2D] = InstanceSegmentation2D  # noqa: F811
    SemanticSegmentation3D: Type[SemanticSegmentation3D] = SemanticSegmentation3D  # noqa: F811
    InstanceSegmentation3D: Type[InstanceSegmentation3D] = InstanceSegmentation3D  # noqa: F811
    OpticalFlow: Type[OpticalFlow] = OpticalFlow  # noqa: F811
    Depth: Type[Depth] = Depth  # noqa: F811
    SurfaceNormals3D: Type[SurfaceNormals3D] = SurfaceNormals3D  # noqa: F811
    SurfaceNormals2D: Type[SurfaceNormals2D] = SurfaceNormals2D  # noqa: F811
    SceneFlow: Type[SceneFlow] = SceneFlow  # noqa: F811
    MaterialProperties2D: Type[MaterialProperties2D] = MaterialProperties2D  # noqa: F811
    Albedo2D: Type[Albedo2D] = Albedo2D  # noqa: F811
    Points2D: Type[Points2D] = Points2D  # noqa: F811
    Polygons2D: Type[Polygons2D] = Polygons2D  # noqa: F811
    Polylines2D: Type[Polylines2D] = Polylines2D  # noqa: F811
    PointCaches: Type[PointCaches] = PointCaches  # noqa: F811
