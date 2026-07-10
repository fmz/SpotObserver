"""
check_graphnav_status.py

Diagnostic: for each robot, check whether a GraphNav map exists, and whether
the robot is currently localized against it. Doesn't modify anything on the
robot -- pure read-only check.
"""

from bosdyn.client.graph_nav import GraphNavClient
from pyspotobserver.connection import SpotConnection
from pyspotobserver.config import SpotConfig


def check_graphnav(robot_ip, username, password, label):
    print(f"--- {label} ({robot_ip}) ---")
    config = SpotConfig(robot_ip=robot_ip, username=username, password=password)
    conn = SpotConnection(config)
    conn.connect()
    try:
        graph_nav_client = conn.robot.ensure_client(GraphNavClient.default_service_name)

        # 1. Does a map exist on this robot right now?
        try:
            graph = graph_nav_client.download_graph()
            n_waypoints = len(graph.waypoints)
            n_edges = len(graph.edges)
            print(f"  Map loaded: {n_waypoints} waypoints, {n_edges} edges")
            has_map = n_waypoints > 0
        except Exception as e:
            print(f"  Failed to download graph: {e}")
            has_map = False

        # 2. Is the robot currently localized against it?
        try:
            state = graph_nav_client.get_localization_state()
            waypoint_id = state.localization.waypoint_id
            if waypoint_id:
                seed_tform_body = state.localization.seed_tform_body
                print(f"  LOCALIZED -- nearest waypoint: {waypoint_id}")
                print(f"  seed_tform_body position: "
                      f"({seed_tform_body.position.x:.3f}, "
                      f"{seed_tform_body.position.y:.3f}, "
                      f"{seed_tform_body.position.z:.3f})")
                is_localized = True
            else:
                print("  NOT LOCALIZED (no waypoint_id in current localization state)")
                is_localized = False
        except Exception as e:
            print(f"  Failed to get localization state: {e}")
            is_localized = False

        return {"has_map": has_map, "is_localized": is_localized}
    finally:
        conn.disconnect()


if __name__ == "__main__":
    ROBOT_A_IP = "128.148.138.21"
    ROBOT_B_IP = "128.148.138.22"
    USERNAME = "user"
    PASSWORD = "bigbubbabigbubba"

    result_a = check_graphnav(ROBOT_A_IP, USERNAME, PASSWORD, "Robot A")
    print()
    result_b = check_graphnav(ROBOT_B_IP, USERNAME, PASSWORD, "Robot B")

    print()
    print("=== Summary ===")
    print(f"Robot A: map={result_a['has_map']}, localized={result_a['is_localized']}")
    print(f"Robot B: map={result_b['has_map']}, localized={result_b['is_localized']}")

    if result_a["has_map"] and result_b["has_map"] and result_a["is_localized"] and result_b["is_localized"]:
        print("Both robots have a map and are localized -- seed_tform_body is usable right now.")
    elif result_a["has_map"] or result_b["has_map"]:
        print("At least one robot has a map, but setup is incomplete -- check upload/localization steps.")
    else:
        print("No map found on either robot -- one needs to be recorded first.")