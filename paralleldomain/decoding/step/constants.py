from paralleldomain.model.class_mapping import ClassDetail

from pd.internal.labels import labels

PD_CLASS_DETAILS_JSON = []

for label in labels:
    label_dict = {
        "name": label.name,
        "id": label.id,
        "color": {"r": label.color[0], "g": label.color[1], "b": label.color[2]},
        "isthing": label.is_thing,
        "supercategory": "",
    }
    PD_CLASS_DETAILS_JSON.append(label_dict)
PD_CLASS_DETAILS_JSON = sorted(PD_CLASS_DETAILS_JSON, key=lambda d: d["name"])


PD_CLASS_DETAILS = [
    ClassDetail(
        name=c["name"],
        id=c["id"],
        instanced=c["isthing"],
        meta=dict(supercategory=c["supercategory"], color=c["color"]),
    )
    for c in PD_CLASS_DETAILS_JSON
]
