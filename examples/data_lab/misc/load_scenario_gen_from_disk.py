import logging
import sys

from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import SensorConfig, SensorRig
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_png
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation
from paralleldomain.visualization.model_visualization import show_frame

setup_loggers(logger_names=[__name__, "paralleldomain"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.4.1-beta")


sensor_rig = SensorRig(
    sensor_configs=[
        SensorConfig.create_camera_sensor(
            name="Front",
            width=1920,
            height=1080,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
        )
    ]
)

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = r"/path/to/scenario_gen.json"

loaded_scenario = data_lab.Scenario.load_scenario(
    path, sensor_rig=sensor_rig  # Optional: overwrite sensor rig from build-sim-state stage with a custom rig
)

AnyPath("out").mkdir(exist_ok=True)
for frame, scene in data_lab.create_frame_stream(
    scenario=loaded_scenario,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
):
    show_frame(frame=frame)
    for camera_frame in frame.camera_frames:
        write_png(
            obj=camera_frame.image.rgb, path=AnyPath(f"out/{camera_frame.sensor_name}_{camera_frame.frame_id:0>18}.png")
        )
