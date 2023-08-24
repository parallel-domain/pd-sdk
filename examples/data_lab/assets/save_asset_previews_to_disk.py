import json
import logging.config
import os
import sys

import pd.management
from internal.apps.asset_browser import Constants
from pd.assets import ObjAssets, init_asset_registry_version
from pd.util.snapshot import generate_state_for_asset_snap, get_location_for_asset_snap
from tqdm import tqdm

from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_png
from tests.helpers import create_minimal_build_sim_state

"""
Asset images generator

This script generates asset images for all the assets listed in the asset registry.
It generates an RGB image and a Semantic Segmentation annotated image for each asset.
"""

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "brief": {"format": "[%(levelname)s] %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "brief",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "brief",
            "filename": "asset_images.log",
            "mode": "w",
        },
    },
    "loggers": {
        "": {
            "handlers": ["file"],
            "level": "DEBUG",
        },
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()

ASSETS_NAME_FILE = "./out.txt"
OUTPUT_DIR = "./asset_preview_images"
IG_ADDRESS = "ssl://ig.step-api-dev.paralleldomain.com:300X"
SIM_ADDRESS = "ssl://sim.step-api-dev.paralleldomain.com:300X"
IG_VERSION = "v2.4.1-beta"

pd.management.org = os.environ["PD_CLIENT_ORG_ENV"]
pd.management.api_key = os.environ["PD_CLIENT_STEP_API_KEY_ENV"]
client_cert_file = os.environ["PD_CLIENT_CREDENTIALS_PATH_ENV"]
resolution = (1080, 1080)
output_path = AnyPath(OUTPUT_DIR)

if output_path.exists():
    sys.exit(f"Error: Output directory {output_path} already exists. Please specify a different directory.")
output_path.mkdir(exist_ok=True)
rgb_output_path = output_path / "rgb"
output_path.mkdir(exist_ok=True)
rgb_output_path.mkdir(exist_ok=True)

session = pd.session.StepIgSession(request_addr=IG_ADDRESS, client_cert_file=client_cert_file)
sim_session = pd.session.SimSession(request_addr=SIM_ADDRESS, client_cert_file=client_cert_file)

with session, sim_session:
    init_asset_registry_version(IG_VERSION)

    prev_asset_name = None
    sensor_agent_id_overwrite = pd.state.rand_agent_id()

    if ASSETS_NAME_FILE:
        asset_names = []
        with open(ASSETS_NAME_FILE) as file:
            for line in file:
                asset_names.append(line.strip())
        asset_count = len(asset_names)

    else:
        asset_objs = ObjAssets.select(ObjAssets.name).order_by(ObjAssets.name)
        asset_names = map(lambda o: o.name, asset_objs)
        asset_count = asset_objs.count()

    world_time = 0.0

    logger.info("Loading Location... ")
    location = get_location_for_asset_snap()
    time_of_day = Constants.DAY_LIGHTING
    session.load_location(location, time_of_day)

    build_sim_state = create_minimal_build_sim_state(location=location)
    build_sim_state_str = json.dumps(build_sim_state.to_dict(), indent=2)
    sim_session.load_scenario_generation(scenario_gen=build_sim_state_str, location_index=0)

    pbar = tqdm(asset_names, total=asset_count)
    for asset_name in pbar:
        pbar.set_description(f"{asset_name:40s}")

        asset_obj = ObjAssets.get_or_none(ObjAssets.name == asset_name)
        if not asset_obj:
            logger.warning(f"Failed to find asset '{asset_name}'")
            continue

        state = generate_state_for_asset_snap(asset_obj=asset_obj, resolution=resolution, raycast=sim_session.raycast)

        if prev_asset_name != asset_name:
            sensor_agent_id_overwrite = pd.state.rand_agent_id()
        prev_asset_name = asset_name
        sensor_agent = next(a for a in state.agents if isinstance(a, pd.state.SensorAgent))
        sensor_agent.id = sensor_agent_id_overwrite

        # Send message data to server
        for i in range(10):
            session.update_state(state)
            world_time += 0.01
            state.simulation_time_sec = world_time

        # RGB image
        sensor_data = session.query_sensor_data(
            sensor_agent.id, sensor_agent.sensors[0].name, pd.state.SensorBuffer.RGB
        )
        if not (sensor_data.height > 0 and sensor_data.width > 0):
            raise Exception("Failed to query sensor image from IG")
        rgb_data = sensor_data.data_as_rgb
        image_path = rgb_output_path / f"{asset_name}.png"
        write_png(path=image_path, obj=rgb_data)
