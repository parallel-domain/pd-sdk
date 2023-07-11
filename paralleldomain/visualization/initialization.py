from typing import Optional

import rerun as rr

_RERUN_IS_INITIALIZED = False
RERUN_RECORDING_STREAM: Optional[rr.RecordingStream] = None


def initialize_viewer(application_id: str = "PD Viewer", entity_root: str = "world", timeless: bool = False) -> bool:
    """
    Initializes a rerun viewer
    Args:
        application_id: Name of the viewer window

    Returns: True if it was initialized, False if its already running

    """
    global _RERUN_IS_INITIALIZED
    global RERUN_RECORDING_STREAM
    if not _RERUN_IS_INITIALIZED:
        rr.init(
            application_id=application_id,
            default_enabled=True,
            strict=True,
            spawn=True,
        )

        rec: rr.RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]
        rec.connect()
        rr.log_view_coordinates(entity_root, timeless=timeless, xyz="FLU")

        RERUN_RECORDING_STREAM = rec
        _RERUN_IS_INITIALIZED = True
        return True
    return False
