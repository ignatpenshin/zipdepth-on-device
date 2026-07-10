import { colorizeInto } from './colormap.js'

const WEIGHTS_URL = '/models/zipdepth_weights.data'
const graphUrl = (s) => `/models/zipdepth_${s}.onnx`
const PANEL = 480 // display panel size (px) per view

// --- Telegram Mini App bootstrap (harmless in a normal browser) -------------
const tg = window.Telegram?.WebApp
if (tg) { try { tg.ready(); tg.expand() } catch {} }

const $ = (id) => document.getElementById(id)
const video = $('video'), viewCanvas = $('view'), badge = $('badge'), hint = $('hint')
const fpsEl = $('fps'), infEl = $('infms'), epEl = $('ep')
const loader = $('loader'), barFill = $('bar-fill'), barText = $('bar-text')
const recDot = $('rec-dot')
const btnLive = $('btn-live'), btnRec = $('btn-rec'), btnPhotoPick = $('btn-photo-pick'), fileInput = $('file')
const resSel = $('res'), camSel = $('camsel'), zoomField = $('zoom-field'), zoom = $('zoom'), camRow = $('cam-row')
const tabLive = $('tab-live'), tabPhoto = $('tab-photo')
const vctx = viewCanvas.getContext('2d')

let displayMode = 'split'   // 'split' | 'overlay'
let mode = 'live'           // 'live' | 'photo'
let size = parseInt(resSel.value, 10)
let ready = false, busy = false, switching = false
let stream = null, track = null, photoImg = null
let lastTs = 0, fpsAvg = 0, blackFrames = 0

// --- per-resolution buffers -------------------------------------------------
let depthCanvas, dctx, rgba, imgData, chw, pre, pctx
function allocFor(s) {
  depthCanvas = document.createElement('canvas'); depthCanvas.width = s; depthCanvas.height = s
  dctx = depthCanvas.getContext('2d')
  rgba = new Uint8ClampedArray(s * s * 4)
  imgData = new ImageData(rgba, s, s)
  chw = new Float32Array(3 * s * s)
  pre = document.createElement('canvas'); pre.width = s; pre.height = s
  pctx = pre.getContext('2d', { willReadFrequently: true })
}
allocFor(size)

// --- Worker -----------------------------------------------------------------
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
worker.onmessage = (e) => {
  const m = e.data
  if (m.type === 'ready' || m.type === 'model-set') {
    ready = true; switching = false; busy = false
    badge.textContent = m.ep.toUpperCase(); badge.className = 'badge ok'; epEl.textContent = m.ep
    if (m.type === 'ready') { loader.classList.add('hidden'); }
    if (mode === 'photo' && photoImg) runInfer(photoImg, photoImg.naturalWidth, photoImg.naturalHeight)
  } else if (m.type === 'result') {
    busy = false
    if (m.size === size) onDepth(new Float32Array(m.depth), m.ms)
  } else if (m.type === 'skip') {
    busy = false
  } else if (m.type === 'error') {
    badge.textContent = 'error'; badge.className = 'badge err'; showHint('Inference error: ' + m.message); busy = false
  }
}

// --- Model loading with progress -------------------------------------------
async function fetchProgress(url, onProgress) {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`${resp.status} ${url}`)
  const total = +resp.headers.get('content-length') || 0
  const reader = resp.body.getReader()
  const chunks = []; let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value); received += value.length
    onProgress(received, total)
  }
  const out = new Uint8Array(received); let off = 0
  for (const c of chunks) { out.set(c, off); off += c.length }
  return out.buffer
}

async function boot() {
  badge.textContent = 'loading'
  try {
    const weights = await fetchProgress(WEIGHTS_URL, (r, t) => {
      const pct = t ? Math.round((r / t) * 100) : 0
      barFill.style.width = pct + '%'
      barText.textContent = t ? `${pct}%  ·  ${(r / 1e6).toFixed(1)}/${(t / 1e6).toFixed(1)} MB` : `${(r / 1e6).toFixed(1)} MB`
    })
    const graph = await (await fetch(graphUrl(size))).arrayBuffer()
    worker.postMessage({ type: 'init', weights, graph, size }, [weights, graph])
  } catch (e) {
    loader.classList.add('hidden'); showHint('Model load failed: ' + e.message)
  }
}
boot()

// --- Resolution switching ---------------------------------------------------
async function setResolution(s) {
  if (s === size || switching) return
  switching = true; ready = false; badge.textContent = 'switching'
  allocFor(s); size = s
  const graph = await (await fetch(graphUrl(s))).arrayBuffer()
  worker.postMessage({ type: 'set-model', graph, size: s }, [graph])
}
resSel.addEventListener('change', () => setResolution(parseInt(resSel.value, 10)))

// --- Preprocess: center-crop square -> size, /255 CHW -----------------------
function preprocess(source, sw, sh) {
  const s = Math.min(sw, sh), sx = (sw - s) / 2, sy = (sh - s) / 2
  pctx.drawImage(source, sx, sy, s, s, 0, 0, size, size)
  const { data } = pctx.getImageData(0, 0, size, size)
  const plane = size * size
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    chw[i] = data[p] / 255
    chw[i + plane] = data[p + 1] / 255
    chw[i + 2 * plane] = data[p + 2] / 255
  }
  return chw.slice().buffer
}
function runInfer(source, sw, sh) {
  if (!ready || busy) return
  busy = true
  const buf = preprocess(source, sw, sh)
  worker.postMessage({ type: 'infer', buffer: buf, reqSize: size }, [buf])
}

// --- Depth result -> depth canvas + stats -----------------------------------
function onDepth(depth, ms) {
  colorizeInto(depth, rgba, 255)
  dctx.putImageData(imgData, 0, 0)
  infEl.textContent = ms.toFixed(0) + ' ms'
  const now = performance.now()
  if (lastTs) { const f = 1000 / (now - lastTs); fpsAvg = fpsAvg ? fpsAvg * 0.8 + f * 0.2 : f; fpsEl.textContent = fpsAvg.toFixed(0) + ' fps' }
  lastTs = now
  if (mode === 'photo' && photoImg) compose(photoImg, photoImg.naturalWidth, photoImg.naturalHeight)
}

// --- Compositor: draw the view (split | overlay) ----------------------------
function setViewSize(w, h) { if (viewCanvas.width !== w) viewCanvas.width = w; if (viewCanvas.height !== h) viewCanvas.height = h }
function compose(source, sw, sh) {
  const s = Math.min(sw, sh), sx = (sw - s) / 2, sy = (sh - s) / 2
  if (displayMode === 'overlay') {
    setViewSize(PANEL, PANEL)
    vctx.drawImage(source, sx, sy, s, s, 0, 0, PANEL, PANEL)
    vctx.globalAlpha = 0.6; vctx.drawImage(depthCanvas, 0, 0, PANEL, PANEL); vctx.globalAlpha = 1
  } else {
    const portrait = viewCanvas.parentElement.clientHeight >= viewCanvas.parentElement.clientWidth
    if (portrait) {
      setViewSize(PANEL, PANEL * 2)
      vctx.drawImage(source, sx, sy, s, s, 0, 0, PANEL, PANEL)
      vctx.drawImage(depthCanvas, 0, PANEL, PANEL, PANEL)
    } else {
      setViewSize(PANEL * 2, PANEL)
      vctx.drawImage(source, sx, sy, s, s, 0, 0, PANEL, PANEL)
      vctx.drawImage(depthCanvas, PANEL, 0, PANEL, PANEL)
    }
  }
}

// --- Live render + inference loop -------------------------------------------
function tick() {
  requestAnimationFrame(tick)
  if (mode !== 'live' || !stream) return
  if (video.readyState < 2 || !video.videoWidth) return
  compose(video, video.videoWidth, video.videoHeight)
  if (detectBlack()) return
  if (ready && !busy && !switching) runInfer(video, video.videoWidth, video.videoHeight)
}
requestAnimationFrame(tick)

function detectBlack() {
  pctx.drawImage(video, 0, 0, 8, 8)
  const s = pctx.getImageData(0, 0, 8, 8).data
  let sum = 0; for (let i = 0; i < s.length; i += 4) sum += s[i] + s[i + 1] + s[i + 2]
  if (sum / (64 * 3) < 3) {
    if (++blackFrames > 40) {
      stopLive(); switchMode('photo')
      showHint('Camera returned a black frame (known iOS/Telegram issue).\nUse Photo mode instead.')
      return true
    }
  } else blackFrames = 0
  return false
}

// --- Camera: start/stop, device list, native zoom ---------------------------
async function startLive(deviceId) {
  try {
    const video_c = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { ...video_c, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    })
    video.srcObject = stream; await video.play()
    track = stream.getVideoTracks()[0]
    mode = 'live'; blackFrames = 0; hideHint()
    btnLive.textContent = '■ Stop'; btnLive.classList.remove('primary')
    await populateCameras(); setupZoom()
  } catch (e) { showHint('Camera unavailable: ' + e.message + '\nTry Photo mode.') }
}
function stopLive() {
  if (stream) stream.getTracks().forEach((t) => t.stop())
  stream = null; track = null; video.srcObject = null
  btnLive.textContent = '● Live'; btnLive.classList.add('primary')
  if (recorder) stopRec()
}
async function populateCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
  if (devs.length <= 1) { camRow.querySelector('.field').style.display = 'none'; return }
  camRow.querySelector('.field').style.display = ''
  const cur = track?.getSettings?.().deviceId
  camSel.innerHTML = ''
  devs.forEach((d, i) => {
    const o = document.createElement('option'); o.value = d.deviceId
    o.textContent = d.label || `Camera ${i + 1}`
    if (d.deviceId === cur) o.selected = true
    camSel.appendChild(o)
  })
}
camSel.addEventListener('change', () => { stopLive(); startLive(camSel.value) })

function setupZoom() {
  const caps = track?.getCapabilities?.()
  if (caps && caps.zoom && caps.zoom.max > caps.zoom.min) {
    zoomField.hidden = false
    zoom.min = caps.zoom.min; zoom.max = caps.zoom.max
    zoom.step = caps.zoom.step || 0.1
    zoom.value = track.getSettings().zoom || caps.zoom.min
  } else { zoomField.hidden = true }
}
zoom.addEventListener('input', () => {
  if (track) track.applyConstraints({ advanced: [{ zoom: parseFloat(zoom.value) }] }).catch(() => {})
})

// --- Recording (captures whatever the view shows: split or overlay) ---------
let recorder = null, chunks = []
function pickMime() {
  for (const m of ['video/mp4;codecs=h264', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'])
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m
  return ''
}
function startRec() {
  const cs = viewCanvas.captureStream(30)
  const mime = pickMime()
  recorder = new MediaRecorder(cs, mime ? { mimeType: mime } : undefined)
  chunks = []
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }
  recorder.onstop = () => {
    const type = recorder.mimeType || 'video/webm'
    const blob = new Blob(chunks, { type })
    const ext = type.includes('mp4') ? 'mp4' : 'webm'
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `zipdepth_${displayMode}.${ext}`
    document.body.appendChild(a); a.click(); a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 4000)
  }
  recorder.start()
  recDot.classList.remove('hidden'); btnRec.textContent = '■ Stop rec'; btnRec.classList.add('rec-on')
}
function stopRec() {
  try { recorder && recorder.state !== 'inactive' && recorder.stop() } catch {}
  recorder = null; recDot.classList.add('hidden'); btnRec.textContent = 'Record'; btnRec.classList.remove('rec-on')
}
btnRec.addEventListener('click', () => { if (recorder) stopRec(); else if (stream) startRec(); else showHint('Start the camera first.') })

// --- Photo mode -------------------------------------------------------------
function loadPhoto(file) {
  const img = new Image()
  img.onload = () => { photoImg = img; runInfer(img, img.naturalWidth, img.naturalHeight); URL.revokeObjectURL(img.src) }
  img.src = URL.createObjectURL(file)
}
btnPhotoPick.addEventListener('click', () => fileInput.click())
fileInput.addEventListener('change', (e) => { if (e.target.files[0]) loadPhoto(e.target.files[0]) })

// --- Mode + view + modal wiring ---------------------------------------------
function switchMode(m) {
  mode = m
  tabLive.classList.toggle('active', m === 'live'); tabPhoto.classList.toggle('active', m === 'photo')
  const live = m === 'live'
  btnLive.hidden = !live; btnRec.hidden = !live; camRow.hidden = !live
  btnPhotoPick.hidden = live
  if (!live) stopLive()
}
tabLive.addEventListener('click', () => switchMode('live'))
tabPhoto.addEventListener('click', () => switchMode('photo'))
btnLive.addEventListener('click', () => (stream ? stopLive() : startLive()))

$('view-seg').addEventListener('click', (e) => {
  const b = e.target.closest('.seg-btn'); if (!b) return
  displayMode = b.dataset.view
  document.querySelectorAll('#view-seg .seg-btn').forEach((x) => x.classList.toggle('active', x === b))
  if (mode === 'photo' && photoImg) compose(photoImg, photoImg.naturalWidth, photoImg.naturalHeight)
})

$('btn-info').addEventListener('click', () => $('modal').classList.remove('hidden'))
$('modal-close').addEventListener('click', () => $('modal').classList.add('hidden'))
$('modal').addEventListener('click', (e) => { if (e.target.id === 'modal') $('modal').classList.add('hidden') })

function showHint(t) { hint.textContent = t; hint.classList.remove('hidden') }
function hideHint() { hint.classList.add('hidden') }

switchMode('live')
