"""Build tiny ONNX models with the exact IO contract of the real streaming model
(2 + 48 in, 2 + 48 out, growing n_frames axis with a sliding window) so the real
C++ StreamingONNXModel can be exercised without loading the multi-GB export.

Variants:
    (default)  fp32 caches, batch fixed at 1
    --fp16     fp16 caches, fp32 image IO (the hybrid conversion recipe)
    --dyn      symbolic batch dim "B" everywhere, like the dynamic re-export;
               each batch slot is an independent sequence

Each cache i appends a slab filled with the constant float(i), so the C++ harness
can verify that cache[i] really carries layer i -- i.e. that GetOutputValues()
comes back in bind order and nothing is cross-wired. In the --dyn variant the
slab is Expand-ed to the runtime batch, so the check holds per slot.

depth = per-image mean over rgb channels, so output depends on the real input and
batch slots stay distinguishable.
"""
import pathlib as _pl
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

_HERE = _pl.Path(__file__).resolve().parent
_BUILD = _HERE / "build"
_BUILD.mkdir(exist_ok=True)

H, W, HEADS, P, HDIM, WINDOW, LAYERS = 8, 10, 2, 3, 4, 4, 24

CACHE_FP16 = "--fp16" in sys.argv
DYN_BATCH = "--dyn" in sys.argv
CACHE_T = TensorProto.FLOAT16 if CACHE_FP16 else TensorProto.FLOAT
CACHE_NP = np.float16 if CACHE_FP16 else np.float32
BATCH = "B" if DYN_BATCH else 1

name = "stub_stream"
if CACHE_FP16:
    name += "_fp16cache"
if DYN_BATCH:
    name += "_dyn"
OUT = _BUILD / f"{name}.onnx"

inputs, outputs, nodes, inits = [], [], [], []

inputs.append(helper.make_tensor_value_info("rgb", TensorProto.FLOAT, [BATCH, 3, H, W]))
inputs.append(helper.make_tensor_value_info("sparse_depth", TensorProto.FLOAT, [BATCH, 1, H, W]))
outputs.append(helper.make_tensor_value_info("depth", TensorProto.FLOAT, [BATCH, H, W]))
outputs.append(helper.make_tensor_value_info("depth_conf", TensorProto.FLOAT, [BATCH, H, W]))

nodes.append(helper.make_node("ReduceMean", ["rgb"], ["rgb_mean"], axes=[1], keepdims=0))
nodes.append(helper.make_node("Identity", ["rgb_mean"], ["depth"]))
nodes.append(helper.make_node("ReduceMean", ["sparse_depth"], ["sd_mean"], axes=[1], keepdims=0))
nodes.append(helper.make_node("Identity", ["sd_mean"], ["depth_conf"]))

# Slice(concat, starts=[-WINDOW], ends=[huge], axes=[2]) grows then clamps,
# exactly like the real graph's retention window.
inits.append(numpy_helper.from_array(np.array([-WINDOW], dtype=np.int64), "slice_start"))
inits.append(numpy_helper.from_array(np.array([2**31], dtype=np.int64), "slice_end"))
inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "slice_axis"))

if DYN_BATCH:
    # Runtime slab shape [B, HEADS, 1, P, HDIM], derived from rgb's batch dim.
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "b_idx"))
    inits.append(numpy_helper.from_array(
        np.array([HEADS, 1, P, HDIM], dtype=np.int64), "slab_tail"))
    nodes.append(helper.make_node("Shape", ["rgb"], ["rgb_shape"]))
    nodes.append(helper.make_node("Gather", ["rgb_shape", "b_idx"], ["b_1d"], axis=0))
    nodes.append(helper.make_node("Concat", ["b_1d", "slab_tail"], ["slab_shape"], axis=0))

idx = 0
for layer in range(LAYERS):
    for kind in ("k", "v"):
        pin = f"past_{kind}_{layer:02d}"
        pout = f"new_{kind}_{layer:02d}"
        inputs.append(helper.make_tensor_value_info(
            pin, CACHE_T, [BATCH, HEADS, "n_frames", P, HDIM]))
        outputs.append(helper.make_tensor_value_info(
            pout, CACHE_T, [BATCH, HEADS, "n_frames_out", P, HDIM]))

        slab = np.full((1, HEADS, 1, P, HDIM), float(idx), dtype=CACHE_NP)
        inits.append(numpy_helper.from_array(slab, f"slab_{idx}"))
        slab_src = f"slab_{idx}"
        if DYN_BATCH:
            nodes.append(helper.make_node(
                "Expand", [f"slab_{idx}", "slab_shape"], [f"slab_b_{idx}"]))
            slab_src = f"slab_b_{idx}"
        nodes.append(helper.make_node("Concat", [pin, slab_src], [f"cat_{idx}"], axis=2))
        nodes.append(helper.make_node(
            "Slice", [f"cat_{idx}", "slice_start", "slice_end", "slice_axis"], [pout]))
        idx += 1

graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"wrote {OUT}")
print(f"inputs={len(inputs)} outputs={len(outputs)} window={WINDOW} "
      f"geometry H={H} W={W} heads={HEADS} P={P} head_dim={HDIM} "
      f"cache={'FLOAT16' if CACHE_FP16 else 'FLOAT'} batch={BATCH}")
