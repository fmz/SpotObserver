import numpy as np
from .config import CameraType

# Known robot IPs
_GOUGER_IP = "128.148.138.21"
_TUSKER_IP = "128.148.138.22"

# Color correction matrices per robot, keyed by CameraType.
# Convention: corrected_pixel = raw_pixel @ M  (row-vector form),
# applied in numpy as: img @ M  for (H, W, 3) images.
_GOUGER_CCMS: dict = {
    CameraType.LEFT: np.array([
        [ 1.4980662, -0.0707718, -0.0458571],
        [-0.4865953,  0.8866953, -0.4204259],
        [ 0.5937464,  0.3074408,  1.1898727],
    ], dtype=np.float32),
    CameraType.RIGHT: np.array([
        [ 2.1307423,  0.1372727,  0.0647798],
        [ 0.0568109,  1.8951481, -0.4296177],
        [ 0.5981083,  0.3732290,  2.0813310],
    ], dtype=np.float32),
    CameraType.FRONTLEFT: np.array([
        [ 1.4337976, -0.1356816, -0.1389128],
        [-0.3111499,  0.9810063, -0.2869629],
        [ 0.3030030,  0.0814753,  0.9887888],
    ], dtype=np.float32),
    CameraType.FRONTRIGHT: np.array([
        [ 2.9136648, -0.0958534, -0.0561581],
        [-0.5915730,  1.7521565, -0.7878229],
        [ 0.6022132,  0.1194339,  1.6545299],
    ], dtype=np.float32),
    CameraType.BACK: np.array([
        [ 1.9841450, -0.3369173, -0.2812417],
        [ 0.0458503,  2.0369307, -0.5472642],
        [-0.1788291, -0.3260584,  1.9366791],
    ], dtype=np.float32),
    CameraType.HAND: np.array([
        [ 1.1079912,  0.1713922,  0.0659761],
        [ 0.3693360,  0.8800426,  0.0744824],
        [ 0.0590054,  0.0759306,  0.5835285],
    ], dtype=np.float32),
}

_TUSKER_CCMS: dict = {
    CameraType.LEFT: np.array([
        [ 1.8952084, -0.0917113, -0.1345257],
        [-0.2042349,  1.4999812, -0.4616307],
        [ 0.3097906,  0.0590297,  1.4913727],
    ], dtype=np.float32),
    CameraType.RIGHT: np.array([
        [ 1.5771384, -0.0511120, -0.0410739],
        [-0.6573969,  0.9523082, -0.5956711],
        [ 0.3767823,  0.3439774,  1.8481518],
    ], dtype=np.float32),
    CameraType.FRONTLEFT: np.array([
        [ 2.1959698,  0.0050106, -0.0217284],
        [-0.3654362,  1.3668996, -0.4870523],
        [ 0.5744365,  0.2933731,  1.5111261],
    ], dtype=np.float32),
    CameraType.FRONTRIGHT: np.array([
        [ 3.1622671, -0.1799195, -0.1370601],
        [-0.8215924,  2.0136070, -0.9268775],
        [ 0.4902452, -0.0647052,  2.0256366],
    ], dtype=np.float32),
    CameraType.BACK: np.array([
        [ 1.4324871, -0.1230974, -0.1201269],
        [-0.6696954,  0.7845523, -0.4282919],
        [ 0.6980419,  0.5317950,  1.5868878],
    ], dtype=np.float32),
    CameraType.HAND: np.array([
        [ 1.0591727,  0.1546625,  0.0633971],
        [ 0.3795602,  0.8702585,  0.0805920],
        [ 0.0184532,  0.0273186,  0.4979610],
    ], dtype=np.float32),
}

_ROBOT_CCMS: dict = {
    _GOUGER_IP: _GOUGER_CCMS,
    _TUSKER_IP: _TUSKER_CCMS,
}