from datetime import datetime, timedelta
from typing import Dict, List


def frame_id_to_timestamp(frame_id: str) -> datetime:
    """
    frame_id is of the form "xxxxx_imgx.ppm"
    Since there is no true framerate or timestamp in FlyingChairs, we make one up.
    """
    epoch_time = datetime(1970, 1, 1)
    seconds = int(frame_id[:5]) + 0.1 * int(frame_id[9])
    timestamp = epoch_time + timedelta(seconds)
    return timestamp
