import logging

from paralleldomain.model.annotation import AnnotationTypes

ANNOTATION_NAME_TO_CLASS = {
    AnnotationTypes.BoundingBoxes2D.__name__: AnnotationTypes.BoundingBoxes2D,
    AnnotationTypes.BoundingBoxes3D.__name__: AnnotationTypes.BoundingBoxes3D,
    AnnotationTypes.SemanticSegmentation2D.__name__: AnnotationTypes.SemanticSegmentation2D,
    AnnotationTypes.InstanceSegmentation2D.__name__: AnnotationTypes.InstanceSegmentation2D,
    AnnotationTypes.SemanticSegmentation3D.__name__: AnnotationTypes.SemanticSegmentation3D,
    AnnotationTypes.InstanceSegmentation3D.__name__: AnnotationTypes.InstanceSegmentation3D,
    AnnotationTypes.OpticalFlow.__name__: AnnotationTypes.OpticalFlow,
    AnnotationTypes.Depth.__name__: AnnotationTypes.Depth,
    AnnotationTypes.SurfaceNormals3D.__name__: AnnotationTypes.SurfaceNormals3D,
    AnnotationTypes.SurfaceNormals2D.__name__: AnnotationTypes.SurfaceNormals2D,
    AnnotationTypes.SceneFlow.__name__: AnnotationTypes.SceneFlow,
    AnnotationTypes.MaterialProperties2D.__name__: AnnotationTypes.MaterialProperties2D,
    AnnotationTypes.MaterialProperties3D.__name__: AnnotationTypes.MaterialProperties3D,
    AnnotationTypes.Albedo2D.__name__: AnnotationTypes.Albedo2D,
    AnnotationTypes.Points2D.__name__: AnnotationTypes.Points2D,
    AnnotationTypes.Points3D.__name__: AnnotationTypes.Points3D,
    AnnotationTypes.Polygons2D.__name__: AnnotationTypes.Polygons2D,
    AnnotationTypes.Polygons3D.__name__: AnnotationTypes.Polygons3D,
    AnnotationTypes.Polylines2D.__name__: AnnotationTypes.Polylines2D,
    AnnotationTypes.Polylines3D.__name__: AnnotationTypes.Polylines3D,
    AnnotationTypes.PointCaches.__name__: AnnotationTypes.PointCaches,
}
