import { defineConfig } from 'vite'

// WebGPU (the primary path) does not require cross-origin isolation, so we don't
// set COOP/COEP. The ONNX Runtime Web WASM binaries are loaded from a CDN at
// runtime (see src/worker.js) to stay under Cloudflare Pages' 25 MiB/file limit;
// only the 12 MB model is served same-origin from /models.
export default defineConfig({
  server: { host: true },
  preview: { host: true },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
