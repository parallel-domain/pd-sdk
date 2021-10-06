from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.dataset import DatasetMetadataDTO
from paralleldomain.common.dgp.v1.remote import RemoteArtifactDTO


@dataclass_json
@dataclass
class DatasetArtifactsDTO:
    metadata: DatasetMetadataDTO
    artifact: RemoteArtifactDTO
    derived_from: List[str]
