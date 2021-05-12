```python
from paralleldomain import Dataset

DATA = "/home/nisseknudsen/Data/artifacts/9040ecc0-add7-11eb-b7f0-70cf4959843a"

ds = Dataset.from_path(DATA)

ds.scenes.keys()

print(ds.scenes["scene_000000"].frames[0].sensors["camera_04"].annotations.available_annotation_types)
# OR
print(ds.scenes["scene_000000"].sensors["camera_04"].frames[0].annotations.available_annotation_types)

# only Type 1 (BB3D) currently supported
annotations = ds.scenes["scene_000000"].frames[0].sensors["camera_04"].annotations[1]  # returns map


```