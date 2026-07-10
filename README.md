# ⚡ ZipDepth — on device

Live **monocular depth estimation running 100% in your browser** — WebGPU (with a WASM
fallback) via ONNX Runtime Web. No inference server, no uploads: camera frames never leave
your device. Packaged to run standalone or as a **Telegram Mini App**.

**▶ Live demo:** https://zipdepth.pages.dev

## What it does

- 📷 **Live camera depth** with two views: **Split** (RGB ∣ depth) or **Overlay**.
- 🔍 **Native camera controls** — optical zoom and lens switching (ultra-wide / tele) where
  the device exposes them, not just digital zoom.
- 🎚️ **On-device resolution switch** (128 → 384) to trade quality vs. FPS live.
- ⏺️ **Record** the split or overlay view to a video file.
- 🖼️ **Photo mode** — run depth on a picture (also the automatic fallback if the in-app
  camera misbehaves, e.g. the known iOS/Telegram black-frame bug).
- 📊 Live FPS / inference-time / execution-provider readout.

## How it works

- **Model:** [ZipDepth](https://github.com/fabiotosi92/ZipDepth)'s compact 6.1 M-param
  network, exported to ONNX (FP16). Preprocessing matches the reference `predictor.py`
  exactly: center-crop → `/255` RGB→CHW, **no** ImageNet mean/std. Output is relative
  inverse-depth, min/max-normalized and colorized (inverted Spectral: near = warm).
- **Shared weights, many resolutions:** the 6.1 M weights are identical across input sizes,
  so they're stored **once** in `zipdepth_weights.data` (12 MB, downloaded with a progress
  bar) and every resolution is a tiny ~50 KB graph referencing it via ONNX external data.
  Switching resolution is near-instant — no re-download.
- **Runtime:** `onnxruntime-web` in a Web Worker; WebGPU preferred, single-thread WASM
  fallback. The WASM binaries load from jsDelivr so the deploy stays small.

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173  (camera works on localhost)
```

To test the camera on a phone (needs HTTPS), tunnel it:

```bash
npx cloudflared tunnel --url http://localhost:5173
```

## Deploy (Cloudflare Pages)

```bash
npm run build
npx wrangler pages deploy dist --project-name zipdepth
```

Then point a Telegram bot's Web App URL (via **@BotFather → Menu Button**) at the deployed
site.

## Regenerate the models

The ONNX files in `public/models/` are produced from ZipDepth's `zipdepth_base_npu.pth`
checkpoint. See [`tools/export_web_models.py`](tools/export_web_models.py) — it exports each
resolution with the NPU (unfold-free) upsampler, FP16-quantizes, and consolidates all
weights into one shared external-data file.

## Credits

**Original research — ZipDepth** (ECCV 2026), University of Bologna:
Fabio Tosi · Luca Bartolomei · Matteo Poggi · Stefano Mattoccia.
[Project](https://zipdepth.github.io/) ·
[Paper](https://arxiv.org/abs/2607.08771) ·
[Repo](https://github.com/fabiotosi92/ZipDepth) — MIT licensed.

**This on-device demo** was built for fun by **Ignat Penshin**, right after ZipDepth's
release, to see the model run live on-device in the browser.
[LinkedIn](https://www.linkedin.com/in/ignat-penshin/) ·
[Telegram channel](https://t.me/ignat_sharit) ·
[@ignat_penshin](https://t.me/ignat_penshin)

## License

MIT (this demo). The ZipDepth network and weights are © their authors, released under MIT.
