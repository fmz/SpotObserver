"""Retarget the streaming model's KV-cache retention window.

The window is baked into the graph as 48 Constant nodes feeding the `starts`
input of the Slice that trims each new_* cache output. Rewriting them is a
graph-only edit: the 7.5 GB .onnx.data blob is referenced by filename and is
neither read nor rewritten, so each variant costs ~8 MB and shares the weights.

VRAM saved is linear in the window, and doubled at peak because the graph reads
`past_*` while writing `new_*`:

    per retained frame = 48 * 16 * 1041 * 64 * 4 B = 195.2 MiB   (fp32)
    peak cache         = 2 * window * per_frame

Usage:
    python set_window.py 8
    python set_window.py 8 --in mae_model_step_consolidated.onnx
"""
import argparse, pathlib, sys
import onnx
from onnx import numpy_helper
import numpy as np

HEADS, TOKENS, HEAD_DIM, N_CACHE = 16, 1041, 64, 48
BYTES_PER_FRAME = N_CACHE * HEADS * TOKENS * HEAD_DIM * 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("window", type=int, help="frames of KV cache to retain")
    ap.add_argument("--in", dest="src", default="mae_model_step_consolidated.onnx")
    ap.add_argument("--out", dest="dst", default=None)
    args = ap.parse_args()

    if args.window < 1:
        print("window must be >= 1", file=sys.stderr)
        return 2

    here = pathlib.Path(__file__).parent
    src = here / args.src
    dst = here / (args.dst or f"{src.stem}_w{args.window}.onnx")

    # load_external_data=False keeps the 7.5 GB blob out of memory entirely.
    model = onnx.load(src, load_external_data=False)
    graph = model.graph

    cache_outputs = {v.name for v in graph.output if v.name.startswith("new_")}
    starts_inputs = {
        n.input[1] for n in graph.node
        if n.op_type == "Slice" and any(o in cache_outputs for o in n.output)
    }
    if len(starts_inputs) != N_CACHE:
        print(f"expected {N_CACHE} cache Slice nodes, found {len(starts_inputs)}", file=sys.stderr)
        return 1

    patched, old_values = 0, set()
    new_val = np.array([-args.window], dtype=np.int64)

    for node in graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] in starts_inputs:
            for attr in node.attribute:
                if attr.name == "value":
                    old_values.add(int(numpy_helper.to_array(attr.t)[0]))
                    attr.t.CopyFrom(numpy_helper.from_array(new_val, attr.t.name))
                    patched += 1
    for init in graph.initializer:
        if init.name in starts_inputs:
            old_values.add(int(numpy_helper.to_array(init)[0]))
            init.CopyFrom(numpy_helper.from_array(new_val, init.name))
            patched += 1

    if patched != N_CACHE:
        print(f"patched {patched} of {N_CACHE} window constants -- aborting", file=sys.stderr)
        return 1

    onnx.checker.check_model(model)
    onnx.save(model, dst)

    old = ", ".join(str(-v) for v in sorted(old_values))
    peak = 2 * args.window * BYTES_PER_FRAME / (1024 ** 3)
    print(f"patched {patched} window constants: {old} -> {args.window}")
    print(f"wrote {dst.name} ({dst.stat().st_size / 1024**2:.1f} MB; shares {src.stem}.onnx.data)")
    print(f"fp32 peak KV cache at window {args.window}: {peak:.2f} GiB "
          f"(steady {peak/2:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
