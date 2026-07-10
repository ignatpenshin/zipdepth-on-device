// Inference worker: ONNX Runtime Web, off the UI thread.
// Loads the 12 MB weights once, then hot-swaps tiny per-resolution graphs that
// all reference the same shared external weights file.
import * as ort from 'onnxruntime-web/webgpu'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/'
ort.env.wasm.numThreads = 1

let weights = null       // Uint8Array, shared across resolutions
let session = null
let size = 256
let ep = 'wasm'
let inName = 'image', outName = 'depth'

async function makeSession(graphBuf) {
  const tryOrder = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm']
  let lastErr
  for (const provider of tryOrder) {
    try {
      const s = await ort.InferenceSession.create(new Uint8Array(graphBuf), {
        executionProviders: [provider],
        graphOptimizationLevel: 'all',
        externalData: [{ path: 'zipdepth_weights.data', data: weights }],
      })
      if (session) { try { await session.release() } catch {} }
      session = s
      ep = provider
      inName = s.inputNames[0]; outName = s.outputNames[0]
      return
    } catch (e) { lastErr = e }
  }
  throw lastErr
}

async function infer(buf, reqSize) {
  if (!session || reqSize !== size) { postMessage({ type: 'skip' }); return }
  const input = new ort.Tensor('float32', new Float32Array(buf), [1, 3, size, size])
  const t0 = performance.now()
  const out = await session.run({ [inName]: input })
  const ms = performance.now() - t0
  const d = out[outName].data
  const arr = d instanceof Float32Array ? d : Float32Array.from(d)
  postMessage({ type: 'result', depth: arr.buffer, ms, ep, size }, [arr.buffer])
}

onmessage = async (e) => {
  const m = e.data
  try {
    if (m.type === 'init') {
      weights = new Uint8Array(m.weights)
      size = m.size
      await makeSession(m.graph)
      postMessage({ type: 'ready', ep, size })
    } else if (m.type === 'set-model') {
      size = m.size
      await makeSession(m.graph)
      postMessage({ type: 'model-set', ep, size })
    } else if (m.type === 'infer') {
      await infer(m.buffer, m.reqSize)
    }
  } catch (err) {
    postMessage({ type: 'error', message: String((err && err.message) || err) })
  }
}
