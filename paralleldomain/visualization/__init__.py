import os
import uuid
from typing import Any, Dict, Optional, Tuple

import rerun as rr

_RERUN_IS_INITIALIZED = dict()
RERUN_RECORDING_STREAM: Dict[str, rr.RecordingStream] = dict()
ACTIVE_RECORDING_ID: Optional[str] = None
ACTIVE_APPLICATION_ID: Optional[str] = None


def initialize_viewer(
    recording_id: Optional[str] = None,
    application_id: str = "PD Viewer",
    entity_root: str = "world",
    timeless: bool = False,
) -> bool:
    """Initializes a rerun viewer. This should be called before any rerun logging calls.

    Args:
        application_id: Name of the viewer window

    Returns: `True` if it was initialized, `False` if its already running
    """
    og_id = recording_id
    global ACTIVE_APPLICATION_ID
    global ACTIVE_RECORDING_ID
    global _RERUN_IS_INITIALIZED
    global RERUN_RECORDING_STREAM
    if recording_id not in _RERUN_IS_INITIALIZED or not _RERUN_IS_INITIALIZED[recording_id]:
        recording_id = recording_id if recording_id is not None else str(uuid.uuid4())
        rr.init(
            application_id=application_id,
            recording_id=recording_id,
            default_enabled=True,
            strict=True,
        )
        ACTIVE_RECORDING_ID = recording_id
        ACTIVE_APPLICATION_ID = application_id

        rec: rr.RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]

        if "PD_HEADLESS" in os.environ:
            rr.serve(recording=rec, open_browser=False)
        else:
            rr.spawn(recording=rec)

        rr.log(
            entity_root,
            rr.ViewCoordinates(xyz=rr.components.ViewCoordinates(coordinates=[5, 4, 1])),  # FLU
            timeless=timeless,
        )

        RERUN_RECORDING_STREAM[og_id] = rec
        _RERUN_IS_INITIALIZED[og_id] = True
        return True
    return _RERUN_IS_INITIALIZED[og_id]


def get_active_recording_and_application_ids() -> Tuple[Optional[str], Optional[str]]:
    global ACTIVE_APPLICATION_ID
    global ACTIVE_RECORDING_ID
    return ACTIVE_RECORDING_ID, ACTIVE_APPLICATION_ID


def set_active_recording_and_application_ids(recording_id: str, application_id: str):
    global ACTIVE_APPLICATION_ID
    global ACTIVE_RECORDING_ID
    ACTIVE_APPLICATION_ID = application_id
    ACTIVE_RECORDING_ID = recording_id
