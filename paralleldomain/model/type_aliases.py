from enum import Enum

FrameId = str
SensorName = str
SceneName = str

AnnotationIdentifier = str


class AnnotationType(Enum):
    BoundingBox2D = 0
    BoundingBox3D = 1
    SemanticSegmentation2D = 2