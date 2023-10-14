from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.image import Image

TYPE_TO_FILE_FORMAT = {
    Image: "png",
    AnnotationTypes.BoundingBoxes2D: "pb.json",
    AnnotationTypes.BoundingBoxes3D: "pb.json",
    AnnotationTypes.SemanticSegmentation2D: "png",
    AnnotationTypes.InstanceSegmentation2D: "png",
    AnnotationTypes.OpticalFlow: "png",
    AnnotationTypes.BackwardOpticalFlow: "png",
    AnnotationTypes.Depth: "npz",
    AnnotationTypes.SurfaceNormals2D: "png",
    AnnotationTypes.Points2D: "pb.json",
    AnnotationTypes.Points3D: "pb.json",
    AnnotationTypes.Albedo2D: "png",
    AnnotationTypes.MaterialProperties2D: "png",
    AnnotationTypes.MaterialProperties3D: "npz",
    AnnotationTypes.SurfaceNormals3D: "npz",
}
