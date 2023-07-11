import time
from collections import defaultdict
from contextlib import suppress
from typing import Callable, Dict, Tuple

import flatbuffers
import numpy as np
import pd.internal.fb.generated.python.pdAck as pdAck
import pd.internal.fb.generated.python.pdLoadLocation as pdLoadLocation
import pd.internal.fb.generated.python.pdMessage as pdMessage
import pd.internal.fb.generated.python.pdReturnSensorData as pdReturnSensorData
import pd.internal.fb.generated.python.pdReturnSystemInfo as pdReturnSystemInfo
import pd.internal.fb.generated.python.pdUpdateState as pdUpdateState
import rerun as rr
import ujson
import zmq
from pd.assets import ObjAssets
from pd.data_lab.context import load_map, setup_datalab
from pd.internal.fb.generated.python.pdMessageType import pdMessageType
from pd.state import ModelAgent, State, VehicleAgent
from pd.state.serialize import bytes_to_state

from paralleldomain.data_lab import Location
from paralleldomain.utilities.transformation import Transformation
from paralleldomain.visualization.model_visualization import initialize_viewer

try:
    from itertools import pairwise  # Python 3.10+
except ImportError:
    from itertools import tee

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


class MessageServer:
    def __init__(
        self, location_callback: Callable[[str], None], state_callback: Callable[[State], None], port: int = 9005
    ):
        # default to port 9005
        self._port = port

        # create our zeromq context and socket and bind them to the above port
        context = zmq.Context()
        self._socket = context.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{self._port}")

        # store message handlers for the messages that we currently handle
        self._msg_handlers: Dict[int, Callable] = {
            pdMessageType.pdLoadLocation: self._handle_load_location,
            pdMessageType.pdUpdateState: self._handle_update_state,
            pdMessageType.pdQuerySystemInfo: self._handle_query_system_info,
            pdMessageType.pdQuerySensorData: self._handle_query_sensor_data,
        }

        self._location_callback = location_callback
        self._state_callback = state_callback

    # helper function for sending messages out via zeromq
    def _send_message(self, builder, msg_bytes, msg_type):
        pdMessage.pdMessageStart(builder)
        pdMessage.pdMessageAddMessageType(builder, msg_type)
        pdMessage.pdMessageAddMessage(builder, msg_bytes)
        msg = pdMessage.pdMessageEnd(builder)

        builder.Finish(msg)
        message = builder.Output()

        self._socket.send(message, flags=zmq.DONTWAIT)

    # helper function for sending ack messages (needed after receiving any message)
    def _send_ack(self, ack_msg, ack_msg_type):
        builder = flatbuffers.Builder(1024)

        ack_string = builder.CreateString(ack_msg)
        pdAck.pdAckStart(builder)
        pdAck.pdAckAddMessageType(builder, ack_msg_type)
        pdAck.pdAckAddResponse(builder, ack_string)
        _msg = pdAck.pdAckEnd(builder)

        self._send_message(builder, _msg, pdMessageType.pdAck)

    # helper function for handling load location message
    def _handle_load_location(self, pd_msg):
        # parse out the message and query the location name
        load_location_msg = pdLoadLocation.pdLoadLocation()
        load_location_msg.Init(pd_msg.Message().Bytes, pd_msg.Message().Pos)

        location_name = load_location_msg.LocationName().decode()
        print(f"Location: {location_name}")

        # send our ack
        self._send_ack("pdLoadLocation received.", pdMessageType.pdLoadLocation)

        # grab the umd view from our studio interface and load the location
        self._location_callback(location_name)

    # helper function for handling query sensor data message
    def _handle_query_sensor_data(self, pd_msg):
        # stub out returning a simple 1x1 sensor image so the handler scripts don't break
        builder = flatbuffers.Builder(1024)

        pdReturnSensorData.pdReturnSensorDataStartDataVector(builder, 4)
        builder.PrependUint32(0)
        data = builder.EndVector(4)

        pdReturnSensorData.pdReturnSensorDataStart(builder)
        pdReturnSensorData.pdReturnSensorDataAddWidth(builder, 1)
        pdReturnSensorData.pdReturnSensorDataAddHeight(builder, 1)
        pdReturnSensorData.pdReturnSensorDataAddChannel(builder, 4)
        pdReturnSensorData.pdReturnSensorDataAddData(builder, data)
        _msg = pdReturnSensorData.pdReturnSensorDataEnd(builder)

        self._send_message(builder, _msg, pdMessageType.pdReturnSensorData)

    # helper function for handling query system info message
    def _handle_query_system_info(self, pd_msg):
        builder = flatbuffers.Builder(1024)

        pdReturnSystemInfo.pdReturnSystemInfoStart(builder)
        pdReturnSystemInfo.pdReturnSystemInfoAddVersionMajor(builder, 0)
        pdReturnSystemInfo.pdReturnSystemInfoAddVersionMinor(builder, 0)
        pdReturnSystemInfo.pdReturnSystemInfoAddVersionPatch(builder, 0)
        _msg = pdReturnSystemInfo.pdReturnSystemInfoEnd(builder)

        self._send_message(builder, _msg, pdMessageType.pdReturnSystemInfo)

    # helper function for handling update state message
    def _handle_update_state(self, pd_msg):
        update_state_msg = pdUpdateState.pdUpdateState()
        update_state_msg.Init(pd_msg.Message().Bytes, pd_msg.Message().Pos)

        # parse the sim state data out of our message using the Step SDK
        sim_state_bytes = update_state_msg.SimStateAsNumpy().tobytes()
        sim_state = bytes_to_state(sim_state_bytes)

        self._state_callback(sim_state)

        # return our ack
        self._send_ack("Unhandled message", pd_msg.MessageType())

    # generic handler for unknown messages
    def _handle_unknown_message(self, pd_msg):
        self._send_ack("Unhandled message", pd_msg.MessageType())

    # simple non-blocking message loop on main thread
    def update(self):
        # query for message and return if none are available
        try:
            message_bytes = self._socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return

        # parse out our message header
        pd_msg = pdMessage.pdMessage.GetRootAspdMessage(message_bytes, 0)
        msg_type = pd_msg.MessageType()

        self._msg_handlers.get(msg_type, self._handle_unknown_message)(pd_msg)


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def hsv_to_rgb(h: float, s: float, v: float, a: float) -> Tuple[float, float, float, float]:
    if s:
        if h == 1.0:
            h = 0.0
        i = int(h * 6.0)
        f = h * 6.0 - i

        w = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        if i == 0:
            return v, t, w, a
        if i == 1:
            return q, v, w, a
        if i == 2:
            return w, v, t, a
        if i == 3:
            return w, q, v, a
        if i == 4:
            return t, w, v, a
        if i == 5:
            return v, w, q, a
    else:
        return v, v, v, a


class StateVisualizer:
    def __init__(self):
        self._server = MessageServer(location_callback=self.update_map, state_callback=self.update_agents)

    def update_map(self, location_name: str):
        rr.log_view_coordinates(entity_path="world/umd", xyz="RFU", timeless=True)  # X=Right, Y=Down, Z=Forward

        umd_map = load_map(location=Location(name=location_name))

        for lane_segment_id, lane_segment in umd_map.lane_segments.items():
            left_edge = umd_map.edges[lane_segment.left_edge] if lane_segment.left_edge > 0 else None
            right_edge = umd_map.edges[lane_segment.right_edge] if lane_segment.right_edge > 0 else None

            left_edge_np = left_edge.as_polyline().to_numpy()
            rr.log_line_strip(
                entity_path=f"world/umd/lane_segments/{lane_segment.id}/left_edge/{left_edge.id}",
                positions=left_edge_np if left_edge.open else np.vstack([left_edge_np, left_edge_np[0]]),
                timeless=True,
                ext={**ujson.loads(left_edge.user_data), "open": left_edge.open},  # metadata
                color=[0, 150, 255],
            )
            right_edge_np = right_edge.as_polyline().to_numpy()
            rr.log_line_strip(
                entity_path=f"world/umd/lane_segments/{lane_segment.id}/right_edge/{right_edge.id}",
                positions=right_edge_np if right_edge.open else np.vstack([right_edge_np, right_edge_np[0]]),
                timeless=True,
                ext={**ujson.loads(right_edge.user_data), "open": right_edge.open},  # metadata
                color=[0, 150, 255],
            )

            reference_line = umd_map.edges[lane_segment.reference_line] if lane_segment.reference_line > 0 else None
            reference_line_np = reference_line.as_polyline().to_numpy()
            rr.log_line_strip(
                entity_path=f"world/umd/lane_segments/{lane_segment.id}/reference_line/{reference_line.id}",
                positions=(
                    reference_line_np if reference_line.open else np.vstack([reference_line_np, reference_line_np[0]])
                ),
                timeless=True,
                ext={**ujson.loads(reference_line.user_data), "open": reference_line.open},  # metadata
                color=[255, 195, 0],
            )

        # # optionally create annotations
        # for road_segment_id, road_segment in umd_map.road_segments.items():
        #     ...

        min_area_id, max_area_id = min(umd_map.areas), max(umd_map.areas)
        min_area_id = max(min_area_id, 0)
        max_area_id = min(max_area_id, 1)
        for area_id, area in umd_map.areas.items():
            for edge_id in area.edges:
                edge = umd_map.edges[edge_id]
                edge_np = edge.as_polyline().to_numpy()
                area_user_data = {}  # noqa: F841
                with suppress(ujson.JSONDecodeError):
                    area_user_data = ujson.loads(area.user_data)

                rr.log_line_strip(
                    entity_path=f"world/umd/areas/{area.id}/edges/{edge.id}",
                    positions=edge_np if edge.open else np.vstack([edge_np, edge_np[0]]),
                    timeless=True,
                    ext={
                        **ujson.loads(edge.user_data),
                        **area_user_data,
                        "type": area.type,
                        "height": area.height,
                        "floors": area.floors,
                    },  # metadata
                    color=hsv_to_rgb(
                        h=((area.id - min_area_id) / (max_area_id - min_area_id)),
                        s=1.0,
                        v=1.0,
                        a=1.0,
                    ),
                )

        min_junction_id, max_junction_id = min(umd_map.junctions), max(umd_map.junctions)
        min_junction_id = max(min_junction_id, 0)
        max_junction_id = min(max_junction_id, 1)
        for junction_id, junction in umd_map.junctions.items():
            for edge_id in junction.corners:
                edge = umd_map.edges[edge_id]
                edge_np = edge.as_polyline().to_numpy()

                junction_user_data = {}  # noqa: F841
                with suppress(ujson.JSONDecodeError):
                    junction_user_data = ujson.loads(junction.user_data)
                rr.log_line_strip(
                    entity_path=f"world/umd/junctions/{junction.id}/corners/{edge.id}",
                    positions=edge_np if edge.open else np.vstack([edge_np, edge_np[0]]),
                    timeless=True,
                    ext={
                        **ujson.loads(edge.user_data),
                        **junction_user_data,
                    },  # metadata
                    color=hsv_to_rgb(
                        h=((junction.id - min_junction_id) / (max_junction_id - min_junction_id)),
                        s=1.0,
                        v=1.0,
                        a=1.0,
                    ),
                )

    def update_agents(self, sim_state: State):
        rr.log_view_coordinates(entity_path="world/sim/agents", xyz="RFU", timeless=True)  # X=Right, Y=Down, Z=Forward
        rr.set_time_seconds(timeline="simulation_time", seconds=sim_state.simulation_time_sec)

        agents = [a for a in sim_state.agents if isinstance(a, ModelAgent) or isinstance(a, VehicleAgent)]

        agent_names = [a.asset_name if isinstance(a, ModelAgent) else a.vehicle_type for a in agents]
        assets = ObjAssets.select(ObjAssets.name, ObjAssets.width, ObjAssets.length, ObjAssets.height).where(
            ObjAssets.name.in_(agent_names)
        )
        asset_dimensions_by_name = {obj.name: (obj.width, obj.length, obj.height) for obj in assets}

        for agent in agents:
            metadata = {}
            if isinstance(agent, ModelAgent):
                metadata = {k: getattr(agent, k) for k in ("asset_name",)}
            elif isinstance(agent, VehicleAgent):
                metadata = {
                    k: getattr(agent, k) for k in ("vehicle_type", "vehicle_color", "vehicle_accessory", "is_parked")
                }

            pose = Transformation.from_transformation_matrix(mat=agent.pose, approximate_orthogonal=True)
            rr.log_obb(
                entity_path=f"world/sim/agents/{agent.id}",
                half_size=list(
                    map(
                        lambda x: x / 2,
                        asset_dimensions_by_name.get(
                            agent.asset_name if isinstance(agent, ModelAgent) else agent.vehicle_type,
                            (1.0, 1.0, 1.0),  # default if asset was not found
                        ),
                    )
                ),
                position=pose.translation,
                rotation_q=[
                    pose.quaternion.x,
                    pose.quaternion.y,
                    pose.quaternion.z,
                    pose.quaternion.w,
                ],
                ext=metadata,
                timeless=False,
            )

    def run(self):
        _ = initialize_viewer(application_id="PD State Visualizer")
        rr.log_view_coordinates(entity_path="world", up="+Z", timeless=True)

        while True:
            self._server.update()
            time.sleep(0.005)


def main():
    setup_datalab("local")

    state_visualizer = StateVisualizer()
    state_visualizer.run()


if __name__ == "__main__":
    main()
