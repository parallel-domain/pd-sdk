from datetime import datetime, timezone
from typing import List, Optional, Union

import numpy as np
from google.protobuf import timestamp_pb2


def timestamp_to_datetime(ts: timestamp_pb2.Timestamp) -> datetime:
    return ts.ToDatetime().replace(tzinfo=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> timestamp_pb2.Timestamp:
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(dt)
    return ts


def rec2array(rec: np.ndarray, fields: Optional[Union[List[str], str]] = None):
    """Convert a record/structured array into an ndarray with a homogeneous data type."""
    simplify = False
    if fields is None:
        fields = rec.dtype.names
    elif isinstance(fields, str):
        fields = [fields]
        simplify = True

    # Concatenate fields into an array
    arr = np.vstack([rec[field] for field in fields]).T

    if simplify:
        # remove last dimension (will be of size 1)
        arr = np.squeeze(arr, axis=-1)
    return arr
