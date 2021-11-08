from datetime import timezone
from typing import List, Optional

import numpy as np
from google.protobuf import timestamp_pb2


def timestamp_to_datetime(ts: timestamp_pb2.Timestamp):
    return ts.ToDatetime().replace(tzinfo=timezone.utc)


def rec2array(rec: np.ndarray, fields: Optional[List[str]] = None):
    """Convert a record/structured array into an ndarray with a homogeneous data type."""
    simplify = False
    if fields is None:
        fields = rec.dtype.names
    elif isinstance(fields, str):
        fields = [fields]
        simplify = True
    # Creates a copy and casts all data to the same type
    arr = np.dstack([rec[field] for field in fields])
    # Check for array-type fields. If none, then remove outer dimension.
    # Only need to check first field since np.dstack will anyway raise an
    # exception if the shapes don't match
    # np.dstack will also fail if fields is an empty list
    if not rec.dtype[fields[0]].shape:
        arr = arr[0]
    if simplify:
        # remove last dimension (will be of size 1)
        arr = arr.reshape(arr.shape[:-1])
    return arr
