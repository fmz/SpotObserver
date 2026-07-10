import struct
import numpy as np
from scipy.spatial.transform import Rotation
from pyspotobserver.icp_align import kabsch, icp, compute_initial_positions

DATA_DIR = "/Users/adannaobuba/Documents/Brown/GHOST/GHOST/Assets/PointClouds"

def load_mesh_array(path, width=640, height=480):
    with open(path, "rb") as f:
        data = f.read()
    length = struct.unpack("<i", data[:4])[0]
    assert length == width * height
    return np.frombuffer(data[4:4 + length * 4], dtype="<f4").reshape(height, width)

def backproject(depth, fx, fy, cx, cy):
    h, w = depth.shape
    v, u = np.indices((h, w))
    valid = depth > 0
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)   # camera-frame points, Nx3

def real_pile():
    depth = load_mesh_array(f"{DATA_DIR}/mesh_array_0")
    fx = fy = 640
    print("Backprojecting depth to 3D points...")
    return backproject(depth, fx, fy, 320, 240)

def test_kabsch_recovers_known_transform(real_pile):
    R_true = Rotation.from_euler('xyz', [3, -5, 10], degrees=True).as_matrix()
    t_true = np.array([0.05, -0.02, 0.1])
    rng = np.random.default_rng(0)
    target = real_pile @ R_true.T + t_true
    target += rng.normal(scale=0.005, size=target.shape)
    R_est, t_est = kabsch(real_pile, target)

    print("test_kabsch_recovers_known_transform: True rotation:\n", R_true)
    print("test_kabsch_recovers_known_transform: Estimated rotation:\n", R_est)
    print("test_kabsch_recovers_known_transform: True translation:\n", t_true)
    print("test_kabsch_recovers_known_transform: Estimated translation:\n", t_est)

    assert np.allclose(R_est, R_true, atol=0.01)
    assert np.allclose(t_est, t_true, atol=0.01)

def test_kabsch_rejects_reflection(real_pile):
    mirror = real_pile.copy()
    mirror[:, 0] *= -1          # a reflection, not a rotation
    R_est, _ = kabsch(real_pile, mirror)

    print("test_kabsch_rejects_reflection: Estimated rotation:\n", R_est)
    assert np.linalg.det(R_est) > 0   # must stay a proper rotation

    assert np.isclose(np.linalg.det(R_est), 1.0, atol=1e-6)   # must stay a proper rotation

def test_kabsch_minimum_points():
    source = np.array([[1., 0., 0.], [0., 1., 0.], [-1., -1., 0.]])
    R_true = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    target = source @ R_true.T
    R_est, t_est = kabsch(source, target)

    print("test_kabsch_minimum_points: True rotation:\n", R_true)
    print("test_kabsch_minimum_points: Estimated rotation:\n", R_est)

    assert np.allclose(R_est, R_true, atol=1e-8)

def test_icp_converges_without_given_correspondences(real_pile):
    R_true = Rotation.from_euler('xyz', [3, -5, 10], degrees=True).as_matrix()
    t_true = np.array([0.05, -0.02, 0.1])
    rng = np.random.default_rng(1)
    target = real_pile @ R_true.T + t_true
    target += rng.normal(scale=0.005, size=target.shape)
    R_est, t_est = icp(real_pile, target, max_distance=0.1)

    print("test_icp_converges_without_given_correspondences: True rotation:\n", R_true)
    print("test_icp_converges_without_given_correspondences: Estimated rotation:\n", R_est)
    print("test_icp_converges_without_given_correspondences: True translation:\n", t_true)
    print("test_icp_converges_without_given_correspondences: Estimated translation:\n", t_est)

    assert np.allclose(R_est, R_true, atol=0.05)
    assert np.allclose(t_est, t_true, atol=0.05)

def test_icp_has_no_fitness_signal_on_mismatched_frames(real_pile):
    """Documents a real gap: icp() can't currently tell a good fit from a bad
    one. Two genuinely unrelated real frames still return a confident-looking
    transform. Once icp() returns a fitness/inlier count, tighten this into an
    assertion that fitness is low."""
    depth1 = load_mesh_array(f"{DATA_DIR}/mesh_array_1")
    unrelated = backproject(depth1, 640, 640, 320, 240)
    R_est, t_est = icp(real_pile, unrelated, max_distance=0.1)

    print("test_icp_has_no_fitness_signal_on_mismatched_frames: Estimated rotation:\n", R_est)
    print("test_icp_has_no_fitness_signal_on_mismatched_frames: Estimated translation:\n", t_est)
    print("test_icp_has_no_fitness_signal_on_mismatched_frames: True rotation:\n", R_true)
    print("test_icp_has_no_fitness_signal_on_mismatched_frames: True translation:\n", t_true)

    assert R_est.shape == (3, 3)   # it runs, but there's no correctness signal here yet

depth = load_mesh_array(f"{DATA_DIR}/mesh_array_0")
w, h = 640, 480
fx = fy = max(w, h)
cx, cy = w / 2, h / 2

pile_a = backproject(depth, fx, fy, cx, cy)

# Manufacture "robot B" by applying a KNOWN transform to a copy of the real cloud
R_true = Rotation.from_euler('xyz', [3, -5, 10], degrees=True).as_matrix()  # small, plausible odometry drift
t_true = np.array([0.05, -0.02, 0.1])
rng = np.random.default_rng(0)
pile_b = pile_a @ R_true.T + t_true
pile_b += rng.normal(scale=0.005, size=pile_b.shape)  # a little sensor noise

R_est, t_est = kabsch(pile_a, pile_b)

print("test_kabsch_recovers_known_transform: True rotation:\n", R_true)
print("test_kabsch_recovers_known_transform: Estimated rotation:\n", R_est)
print("test_kabsch_recovers_known_transform: True translation:\n", t_true)
print("test_kabsch_recovers_known_transform: Estimated translation:\n", t_est)

test_kabsch_recovers_known_transform(pile_a)
test_kabsch_rejects_reflection(pile_a)
test_kabsch_minimum_points()
test_icp_converges_without_given_correspondences(pile_a)
test_icp_has_no_fitness_signal_on_mismatched_frames(pile_a)

assert np.allclose(R_est, R_true, atol = .02)
assert np.allclose(t_est, t_true, atol = .02)