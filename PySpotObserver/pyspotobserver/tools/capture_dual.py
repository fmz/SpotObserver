import json
from pathlib import Path
import numpy as np
import cv2
from pyspotobserver.connection import SpotConnection
from pyspotobserver.config import SpotConfig, CameraType
from pyspotobserver.stitch import extract_stitch_params

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import get_vision_tform_body

CAPTURES_DIR = Path(__file__).parent / "captures"

def capture_one_robot(robot_ip, username, password,
                       cameras=CameraType.FRONTLEFT | CameraType.FRONTRIGHT):
    
    config = SpotConfig(robot_ip = robot_ip, username = username, password = password)
    conn = SpotConnection(config)
    conn.connect()

    stream = conn.create_cam_stream()
    stream.start_streaming(cameras)

    try:
        pose = get_robot_pose(conn)

        rgb_images, depth_images = stream.get_current_images(timeout=5.0, copy=True)
        camera_order = stream.get_camera_order()   # e.g. [FRONTLEFT, FRONTRIGHT], matches array order above

        from bosdyn.client.image import build_image_request
        from bosdyn.api import image_pb2
        params_list = []
        for cam in camera_order:
            source = CameraType.get_source_name(cam, depth=False)
            req = build_image_request(source, image_format=image_pb2.Image.FORMAT_JPEG,
                                       pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)
            response = conn.image_client.get_image([req])[0]
            params_list.append(extract_stitch_params(response))
    finally:
        stream.stop_streaming()
        conn.disconnect()
    return rgb_images, depth_images, camera_order, params_list, pose

def backproject_to_body(depth, params):
    h, w = depth.shape
    v, u = np.indices((h, w))
    valid = depth > 0
    z = depth[valid]
    x = (u[valid] - params.cx) * z / params.fx
    y = (v[valid] - params.cy) * z / params.fy
    cam_pts = np.stack([x, y, z], axis=-1)
    return cam_pts @ params.rot.T + params.trans

def get_robot_pose(conn):

    # state_client = conn.robot.ensure_client(RobotStateClient.default_service_name)
    # state = state_client.get_robot_state()

    snapshot = conn.robot.get_frame_tree_snapshot()
    return get_vision_tform_body(snapshot).to_matrix()

def save_capture(name, rgb_images, depth_images, camera_order, params_list, pose):
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    all_points = []
    for cam, rgb, depth, params in zip(camera_order, rgb_images, depth_images, params_list):
        tag = cam.name.lower()  # e.g. "frontleft", "frontright"
        np.save(CAPTURES_DIR / f"{name}_{tag}_depth.npy", depth)
        cv2.imwrite(str(CAPTURES_DIR / f"{name}_{tag}_rgb.png"),
                    cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        with open(CAPTURES_DIR / f"{name}_{tag}_calib.json", "w") as f:
            json.dump({"fx": params.fx, "fy": params.fy, "cx": params.cx, "cy": params.cy,
                       "rot": params.rot.tolist(), "trans": params.trans.tolist()}, f, indent=2)
        all_points.append(backproject_to_body(depth, params))

    combined = np.concatenate(all_points, axis=0)
    np.save(CAPTURES_DIR / f"{name}_points_body.npy", combined)
    np.save(CAPTURES_DIR / f"{name}_pose.npy", pose)
    print(f"Saved {name}: {len(camera_order)} cameras, {combined.shape[0]} combined points, "
          f"pose (vision_tform_body) -> {CAPTURES_DIR}")

if __name__ == "__main__":
    ROBOT_A_IP = "128.148.138.21"
    ROBOT_B_IP = "128.148.138.22"
    USERNAME = "user"
    PASSWORD = "bigbubbabigbubba"

    rgb_a, depth_a, order_a, params_a, pose_a = capture_one_robot(ROBOT_A_IP, USERNAME, PASSWORD)
    save_capture("robotA", rgb_a, depth_a, order_a, params_a, pose_a)

    rgb_b, depth_b, order_b, params_b, pose_b = capture_one_robot(ROBOT_B_IP, USERNAME, PASSWORD)
    save_capture("robotB", rgb_b, depth_b, order_b, params_b, pose_b)