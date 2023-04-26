import logging.config
import os
import sys
from pathlib import Path

import cv2
import pd.management
from pd.assets import ObjAssets, init_asset_registry_version
from pd.util.snapshot import generate_state_for_asset_snap
from tqdm import tqdm

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
IG_VERSION = "v2.0.0-beta"

pd.management.org = os.environ["PD_CLIENT_ORG_ENV"]
pd.management.api_key = os.environ["PD_CLIENT_STEP_API_KEY_ENV"]
client_cert_file = os.environ["PD_CLIENT_CREDENTIALS_PATH_ENV"]
resolution = (1080, 1080)
output_path = Path(OUTPUT_DIR)

if output_path.exists():
    sys.exit(f"Error: Output directory {output_path} already exists. Please specify a different directory.")
output_path.mkdir(exist_ok=True)
rgb_output_path = output_path / "rgb"
output_path.mkdir(exist_ok=True)
rgb_output_path.mkdir(exist_ok=True)

session = pd.session.StepSession(request_addr=IG_ADDRESS, client_cert_file=client_cert_file)

with session:
    init_asset_registry_version(IG_VERSION)

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
    location_loaded = False
    pbar = tqdm(asset_names, total=asset_count)
    for asset_name in pbar:
        pbar.set_description(f"{asset_name:40s}")

        asset_obj = ObjAssets.get_or_none(ObjAssets.name == asset_name)
        if not asset_obj:
            logger.warning(f"Failed to find asset '{asset_name}'")
            continue

        state = generate_state_for_asset_snap(asset_obj, resolution)
        state.simulation_time_sec = world_time

        if not location_loaded:
            version = session.system_info.version
            location = state.world_info.location
            time_of_day = state.world_info.time_of_day
            session.load_location(location, time_of_day)
            location_loaded = True

        # Send message data to server
        for i in range(10):
            session.update_state(state)
            world_time += 0.01

        # RGB image
        sensor_agent = next(a for a in state.agents if isinstance(a, pd.state.SensorAgent))
        sensor_data = session.query_sensor_data(
            sensor_agent.id, sensor_agent.sensors[0].name, pd.state.SensorBuffer.RGB
        )
        if not (sensor_data.height > 0 and sensor_data.width > 0):
            raise Exception("Failed to query sensor image from IG")
        rgb_data = sensor_data.data_as_rgb
        image_path = rgb_output_path / f"{asset_name}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
