from pd.label_engine import LabelData

from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
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


def decode_class_map(ontology_data: LabelData) -> ClassMap:
    if ontology_data is not None:
        ontology = ontology_data.data_as_semantic_label_map
        semantic_label_map = ontology.semantic_label_map
        pd_class_details = []
        for semantic_id in semantic_label_map:
            c = semantic_label_map[semantic_id]
            class_detail = ClassDetail(
                name=c.label,
                id=int(c.id),
                meta=dict(supercategory="", color={"r": c.color.red, "g": c.color.green, "b": c.color.blue}),
            )
            pd_class_details.append(class_detail)
        class_map = ClassMap(classes=pd_class_details)
        return class_map
    else:
        raise ValueError("Can not decode class_map because ontology_data is empty.")
