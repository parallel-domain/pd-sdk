from datetime import datetime, timedelta
from typing import Dict, List


def frame_id_to_timestamp(frame_id: str) -> datetime:
    epoch_time = datetime(1970, 1, 1)
    # First frame and second frame will be separated by 0.1s, per the 10Hz frame rate in KITTI
    seconds = int(frame_id[:6]) + 0.1 * int(frame_id[7:9])
    timestamp = epoch_time + timedelta(seconds)
    return timestamp
