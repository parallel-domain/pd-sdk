from datetime import timezone

from google.protobuf import timestamp_pb2


def timestamp_to_datetime(ts: timestamp_pb2.Timestamp):
    return ts.ToDatetime().replace(tzinfo=timezone.utc)
