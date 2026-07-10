"""
Export ZipDepth to browser-ready ONNX for this demo.

Produces, in public/models/:
  - zipdepth_weights.data   the 6.1 M FP16 weights, ONCE (identical across resolutions)
  - zipdepth_<S>.onnx       one tiny graph per input size, referencing the shared weights

Run from a checkout of https://github.com/fabiotosi92/ZipDepth (so `zipdepth` and
`scripts/export.py` are importable), with the base_npu checkpoint available:

    pip install torch onnx onnxscript onnxruntime onnxconverter-common
    python export_web_models.py --ckpt checkpoints/zipdepth_base_npu.pth --out ../public/models

Notes:
  - Uses the NPU (unfold-free) upsampler → WebGPU/ONNX-friendly. Requires the *_npu.pth.
  - Forces the legacy TorchScript ONNX exporter (dynamo=False); the dynamo path trips on
    this graph's opset conversion. onnxsim is intentionally skipped (native segfault).
  - Weights are byte-identical across sizes, so all graphs share one external-data file.
"""
import argparse, os, sys, shutil
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16

SIZES = [128, 192, 256, 320, 384]


def export_raw(ckpt, size, tmp_path):
    """Export one fixed-resolution FP32 ONNX using ZipDepth's own export helpers."""
    import torch
    from scripts.export import load_model, export_onnx
    model = load_model(ckpt, "base", "balanced", "cpu", upsample_unfold=False)  # npu variant
    export_onnx(model, (1, 3, size, size), tmp_path)  # writes <tmp>_raw.onnx (onnxsim skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to zipdepth_base_npu.pth")
    ap.add_argument("--out", required=True, help="output dir (this repo's public/models)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) FP32 export + FP16 convert, each with its own temp external-data file.
    for s in SIZES:
        base = os.path.join(args.out, f"zipdepth_{s}")
        export_raw(args.ckpt, s, base + ".onnx")
        raw = base + "_raw.onnx"
        m16 = float16.convert_float_to_float16(onnx.load(raw), keep_io_types=True)
        onnx.save(m16, base + ".onnx", save_as_external_data=True, all_tensors_to_one_file=True,
                  location=f"_w{s}.data", size_threshold=0, convert_attribute=False)
        os.remove(raw)
        if os.path.exists(raw + ".data"):
            os.remove(raw + ".data")

    # 2) Weights are identical across sizes → keep one, point every graph at it.
    shared = os.path.join(args.out, "zipdepth_weights.data")
    os.replace(os.path.join(args.out, f"_w{SIZES[0]}.data"), shared)
    for s in SIZES[1:]:
        os.remove(os.path.join(args.out, f"_w{s}.data"))
    for s in SIZES:
        g = os.path.join(args.out, f"zipdepth_{s}.onnx")
        m = onnx.load(g, load_external_data=False)
        for t in m.graph.initializer:
            for e in t.external_data:
                if e.key == "location":
                    e.value = "zipdepth_weights.data"
        onnx.save(m, g)

    # 3) Validate each graph runs against the shared weights.
    for s in SIZES:
        sess = ort.InferenceSession(os.path.join(args.out, f"zipdepth_{s}.onnx"),
                                    providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": np.random.rand(1, 3, s, s).astype(np.float32)})[0]
        assert out.shape == (1, 1, s, s), out.shape
        print(f"  {s}: OK {out.shape}")
    print(f"shared weights: {os.path.getsize(shared) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
