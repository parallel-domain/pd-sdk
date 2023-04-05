import os
from typing import Iterator

import pd.management
from pd.assets import init_asset_registry_version
from pd.internal.assets.asset_registry import InfoSegmentation, UtilSegmentationCategoriesPanoptic

IG_VERSION = "v2.0.0-beta"
CLASS_NAME = "Debris"

pd.management.org = os.environ["PD_CLIENT_ORG_ENV"]
pd.management.api_key = os.environ["PD_CLIENT_STEP_API_KEY_ENV"]
init_asset_registry_version(IG_VERSION)


def get_all_asset_names() -> Iterator[str]:
    query = InfoSegmentation.select(InfoSegmentation.name)
    for obj in query:
        yield obj.name


def get_all_class_names() -> Iterator[str]:
    query = UtilSegmentationCategoriesPanoptic.select(UtilSegmentationCategoriesPanoptic.name)
    for obj in query:
        yield obj.name


def get_all_assets_in_class(class_name: str) -> Iterator[str]:
    query = (
        InfoSegmentation.select(InfoSegmentation.name)
        .join(UtilSegmentationCategoriesPanoptic)
        .where(UtilSegmentationCategoriesPanoptic.name == class_name)
    )
    for obj in query:
        yield obj.name


# Print out all assets
# for name in get_all_asset_names():
#     print(name)

# Print out all classes
# for name in get_all_class_names():
#     print(name)

# Print out all assets in class
for name in get_all_assets_in_class(CLASS_NAME):
    print(name)
