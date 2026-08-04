"""Build a tiny ONNX model with the exact IO contract of the real streaming model
(2 + 48 in, 2 + 48 out, growing n_frames axis with a sliding window) so the real
C++ StreamingONNXModel can be exercised without loading 7.5 GB.

Each cache i appends a slab filled with the constant float(i), so the C++ harness
can verify that cache[i] really carries layer i -- i.e. that GetOutputValues()
comes back in bind order and nothing is cross-wired.
"""
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_BUILD = _HERE / "build"
_BUILD.mkdir(exist_ok=True)
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

H, W, HEADS, P, HDIM, WINDOW, LAYERS = 8, 10, 2, 3, 4, 4, 24
import sys
CACHE_FP16 = "--fp16" in sys.argv
OUT = (str(_BUILD) + "/"
       + ("stub_stream_fp16cache.onnx" if CACHE_FP16 else "stub_stream.onnx"))
# Hybrid recipe under test: image tensors stay fp32, caches carry fp16.
CACHE_T = TensorProto.FLOAT16 if CACHE_FP16 else TensorProto.FLOAT
CACHE_NP = np.float16 if CACHE_FP16 else np.float32

inputs, outputs, nodes, inits = [], [], [], []

inputs.append(helper.make_tensor_value_info("rgb", TensorProto.FLOAT, [1, 3, H, W]))
inputs.append(helper.make_tensor_value_info("sparse_depth", TensorProto.FLOAT, [1, 1, H, W]))
outputs.append(helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, H, W]))
outputs.append(helper.make_tensor_value_info("depth_conf", TensorProto.FLOAT, [1, H, W]))

# depth = mean over rgb channels, so the output depends on the real input.
nodes.append(helper.make_node("ReduceMean", ["rgb"], ["rgb_mean"], axes=[1], keepdims=0))
nodes.append(helper.make_node("Identity", ["rgb_mean"], ["depth"]))
nodes.append(helper.make_node("ReduceMean", ["sparse_depth"], ["sd_mean"], axes=[1], keepdims=0))
nodes.append(helper.make_node("Identity", ["sd_mean"], ["depth_conf"]))

# Slice(concat, starts=[-WINDOW], ends=[huge], axes=[2]) grows then clamps,
# exactly like the real graph's retention window.
inits.append(numpy_helper.from_array(np.array([-WINDOW], dtype=np.int64), "slice_start"))
inits.append(numpy_helper.from_array(np.array([2**31], dtype=np.int64), "slice_end"))
inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "slice_axis"))

idx = 0
for layer in range(LAYERS):
    for kind in ("k", "v"):
        pin = f"past_{kind}_{layer:02d}"
        pout = f"new_{kind}_{layer:02d}"
        inputs.append(helper.make_tensor_value_info(
            pin, CACHE_T, [1, HEADS, "n_frames", P, HDIM]))
        outputs.append(helper.make_tensor_value_info(
            pout, CACHE_T, [1, HEADS, "n_frames_out", P, HDIM]))

        slab = np.full((1, HEADS, 1, P, HDIM), float(idx), dtype=CACHE_NP)
        inits.append(numpy_helper.from_array(slab, f"slab_{idx}"))
        nodes.append(helper.make_node("Concat", [pin, f"slab_{idx}"], [f"cat_{idx}"], axis=2))
        nodes.append(helper.make_node(
            "Slice", [f"cat_{idx}", "slice_start", "slice_end", "slice_axis"], [pout]))
        idx += 1

graph = helper.make_graph(nodes, "stub_stream", inputs, outputs, initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"wrote {OUT}")
print(f"inputs={len(inputs)} outputs={len(outputs)} window={WINDOW} "
      f"geometry H={H} W={W} heads={HEADS} P={P} head_dim={HDIM} "
      f"cache={'FLOAT16' if CACHE_FP16 else 'FLOAT'}")
