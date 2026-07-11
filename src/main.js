import { colorizeInto } from './colormap.js'
import { CloudRenderer } from './cloud/renderer.js'
import { OrbitCamera } from './cloud/camera.js'
import { IMU } from './sensors.js'

const WEIGHTS_URL = '/models/zipdepth_weights.data'
const graphUrl = (s) => `/models/zipdepth_${s}.onnx`

const $ = (id) => document.getElementById(id)
const video = $('video'), viewCanvas = $('view'), stage = $('stage'), badge = $('badge'), hint = $('hint')
const fpsEl = $('fps'), infEl = $('infms'), epEl = $('ep')
const loader = $('loader'), barFill = $('bar-fill'), barText = $('bar-text')
const recDot = $('rec-dot'), zoomBadge = $('zoom-badge')
const btnLive = $('btn-live'), btnRec = $('btn-rec'), btnPhotoPick = $('btn-photo-pick'), fileInput = $('file')
const resSel = $('res'), camSel = $('camsel'), camRow = $('cam-row')
const zoomRow = $('zoom-row'), zoom = $('zoom'), zoomVal = $('zoomval'), lensBox = $('lens')
const btnCam = $('btn-cam'), camInfo = $('caminfo')
const tabLive = $('tab-live'), tabPhoto = $('tab-photo'), tab3d = $('tab-3d')
const cloudCanvas = $('cloud'), cloudPanel = $('cloud-panel')
const vctx = viewCanvas.getContext('2d')

let displayMode = 'split'   // 'split' | 'overlay'
let mode = 'live'           // 'live' | 'photo'
let size = parseInt(resSel.value, 10)
let ready = false, busy = false, switching = false
let stream = null, track = null, photoImg = null
let lastTs = 0, fpsAvg = 0, blackFrames = 0
let hasNativeZoom = false, zoomFactor = 1     // zoomFactor: digital crop factor (native uses applyConstraints)
let is3D = false, cloud = null, cam = null, imu = null, cloudFrozen = false, cloudRAF = 0

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
    if (m.type === 'ready') loader.classList.add('hidden')
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
  } catch (e) { loader.classList.add('hidden'); showHint('Model load failed: ' + e.message) }
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

// --- Crop (square, center) with digital zoom --------------------------------
function cropRect(sw, sh) {
  const base = Math.min(sw, sh)
  const s = base / (hasNativeZoom ? 1 : zoomFactor)
  return { s, sx: (sw - s) / 2, sy: (sh - s) / 2 }
}

// --- Preprocess: crop -> size, /255 CHW -------------------------------------
let lastCropRGBA = null   // aligned color crop for the in-flight inference (3D)
function preprocess(source, sw, sh) {
  const { s, sx, sy } = cropRect(sw, sh)
  pctx.drawImage(source, sx, sy, s, s, 0, 0, size, size)
  const { data } = pctx.getImageData(0, 0, size, size)
  lastCropRGBA = data
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

// --- Depth result -----------------------------------------------------------
function onDepth(depth, ms) {
  updateStats(ms)
  if (is3D) {
    if (cloud && !cloudFrozen && lastCropRGBA) cloud.setFrame(depth, lastCropRGBA, size)
    return
  }
  colorizeInto(depth, rgba, 255)
  dctx.putImageData(imgData, 0, 0)
  if (mode === 'photo' && photoImg) compose(photoImg, photoImg.naturalWidth, photoImg.naturalHeight)
}
function updateStats(ms) {
  infEl.textContent = ms.toFixed(0) + ' ms'
  const now = performance.now()
  if (lastTs) { const f = 1000 / (now - lastTs); fpsAvg = fpsAvg ? fpsAvg * 0.8 + f * 0.2 : f; fpsEl.textContent = fpsAvg.toFixed(0) + ' fps' }
  lastTs = now
}

// --- Compositor: fills the stage; split or overlay --------------------------
function panel(img, cx, cy, cw, ch, dx, dy, dw, dh) {
  // cover-fit the square crop (cx,cy,cw,ch) into dest rect, clipped
  vctx.save()
  vctx.beginPath(); vctx.rect(dx, dy, dw, dh); vctx.clip()
  const scale = Math.max(dw / cw, dh / ch)
  const w = cw * scale, h = ch * scale
  vctx.drawImage(img, cx, cy, cw, ch, dx + (dw - w) / 2, dy + (dh - h) / 2, w, h)
  vctx.restore()
}
function compose(source, sw, sh) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const W = Math.max(1, Math.round(stage.clientWidth * dpr))
  const H = Math.max(1, Math.round(stage.clientHeight * dpr))
  if (viewCanvas.width !== W) viewCanvas.width = W
  if (viewCanvas.height !== H) viewCanvas.height = H
  vctx.fillStyle = '#000'; vctx.fillRect(0, 0, W, H)
  const { s, sx, sy } = cropRect(sw, sh)
  const dw = depthCanvas.width
  if (displayMode === 'overlay') {
    panel(source, sx, sy, s, s, 0, 0, W, H)
    vctx.globalAlpha = 0.6; panel(depthCanvas, 0, 0, dw, dw, 0, 0, W, H); vctx.globalAlpha = 1
  } else if (H >= W) { // portrait: RGB top, depth bottom
    panel(source, sx, sy, s, s, 0, 0, W, H / 2)
    panel(depthCanvas, 0, 0, dw, dw, 0, H / 2, W, H / 2)
  } else {            // landscape: RGB left, depth right
    panel(source, sx, sy, s, s, 0, 0, W / 2, H)
    panel(depthCanvas, 0, 0, dw, dw, W / 2, 0, W / 2, H)
  }
}

// --- Live render + inference loop -------------------------------------------
function tick() {
  requestAnimationFrame(tick)
  if (mode !== 'live' || !stream) return
  if (video.readyState < 2 || !video.videoWidth) return
  if (!is3D) compose(video, video.videoWidth, video.videoHeight)
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
      showHint('Camera returned a black frame.\nTry Photo mode instead.')
      return true
    }
  } else blackFrames = 0
  return false
}

// --- Camera: start/stop, device list, zoom, info ----------------------------
async function startLive(deviceId) {
  try {
    const v = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { ...v, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
    })
    video.srcObject = stream; await video.play()
    track = stream.getVideoTracks()[0]
    mode = 'live'; blackFrames = 0; hideHint()
    btnLive.textContent = '■ Stop'; btnLive.classList.remove('primary')
    zoomRow.hidden = false; btnCam.hidden = false
    await populateCameras(); setupZoom(); updateCamInfo()
  } catch (e) { showHint('Camera unavailable: ' + e.message + '\nTry Photo mode.') }
}
function stopLive() {
  if (recorder) stopRec()
  if (stream) stream.getTracks().forEach((t) => t.stop())
  stream = null; track = null; video.srcObject = null
  btnLive.textContent = '● Live'; btnLive.classList.add('primary')
  zoomRow.hidden = true; zoomBadge.classList.add('hidden'); btnCam.hidden = true; camInfo.classList.add('hidden')
}
async function populateCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
  camRow.hidden = devs.length <= 1
  const cur = track?.getSettings?.().deviceId
  camSel.innerHTML = ''
  devs.forEach((d, i) => {
    const o = document.createElement('option'); o.value = d.deviceId
    o.textContent = d.label || `Camera ${i + 1}`
    if (d.deviceId === cur) o.selected = true
    camSel.appendChild(o)
  })
}
camSel.addEventListener('change', () => { const id = camSel.value; stopLive(); startLive(id) })

function setupZoom() {
  const caps = track?.getCapabilities?.() || {}
  const set = track?.getSettings?.() || {}
  lensBox.innerHTML = ''
  if (caps.zoom && caps.zoom.max > caps.zoom.min) {
    hasNativeZoom = true
    zoom.min = caps.zoom.min; zoom.max = caps.zoom.max; zoom.step = caps.zoom.step || 0.1
    zoom.value = set.zoom || caps.zoom.min
    buildLensPresets(caps.zoom.min, caps.zoom.max)
  } else {
    hasNativeZoom = false
    zoom.min = 1; zoom.max = 8; zoom.step = 0.1; zoom.value = 1
    buildLensPresets(1, 8)
  }
  applyZoom(parseFloat(zoom.value), false)
}
function buildLensPresets(min, max) {
  const marks = [...new Set([min, 1, 2, 3, 5].filter((x) => x >= min && x <= max))]
  if (marks[marks.length - 1] !== max) marks.push(max)
  marks.forEach((x) => {
    const b = document.createElement('button'); b.className = 'seg-btn'; b.dataset.zoom = x
    b.textContent = (x < 1 ? x.toFixed(1) : x % 1 ? x.toFixed(1) : x) + '×'
    b.addEventListener('click', () => applyZoom(x, true))
    lensBox.appendChild(b)
  })
}
function applyZoom(v, fromButton) {
  const min = parseFloat(zoom.min), max = parseFloat(zoom.max)
  v = Math.min(max, Math.max(min, v))
  zoom.value = v
  zoomVal.textContent = v.toFixed(1) + '×'
  zoomBadge.textContent = v.toFixed(1) + '×'; zoomBadge.classList.toggle('hidden', v === min && !fromButton)
  lensBox.querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('active', Math.abs(parseFloat(b.dataset.zoom) - v) < 0.05))
  if (hasNativeZoom && track) track.applyConstraints({ advanced: [{ zoom: v }] }).catch(() => {})
  else zoomFactor = v
  updateCamInfo()
}
zoom.addEventListener('input', () => applyZoom(parseFloat(zoom.value), false))

// pinch-to-zoom on the stage
let pinchStart = 0, pinchZoom0 = 1
stage.addEventListener('touchstart', (e) => {
  if (e.touches.length === 2) { pinchStart = dist(e.touches); pinchZoom0 = parseFloat(zoom.value) }
}, { passive: true })
stage.addEventListener('touchmove', (e) => {
  if (e.touches.length === 2 && pinchStart) { e.preventDefault(); applyZoom(pinchZoom0 * (dist(e.touches) / pinchStart), true) }
}, { passive: false })
stage.addEventListener('touchend', () => { pinchStart = 0 })
function dist(t) { return Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY) }

function updateCamInfo() {
  if (!track) return
  const s = track.getSettings?.() || {}, c = track.getCapabilities?.() || {}
  const rng = (x) => (x && x.min !== undefined ? `${round(x.min)}–${round(x.max)}` : '')
  const round = (n) => (typeof n === 'number' ? Math.round(n * 100) / 100 : n)
  const lines = [
    ['lens', s.label || camSel.selectedOptions[0]?.textContent || '—'],
    ['resolution', s.width && s.height ? `${s.width}×${s.height}` : '—'],
    ['frame rate', s.frameRate ? `${Math.round(s.frameRate)} fps` : '—'],
    ['facing', s.facingMode || '—'],
    ['zoom', hasNativeZoom ? `${round(s.zoom ?? zoom.value)}× (native ${rng(c.zoom)})` : `${round(zoomFactor)}× (digital)`],
    ['focus', s.focusMode || (c.focusMode ? c.focusMode.join('/') : '—')],
    ['exposure', s.exposureMode || '—'],
    ['torch', c.torch ? 'yes' : 'no'],
  ]
  camInfo.textContent = lines.map(([k, v]) => `${(k + ':').padEnd(12)} ${v}`).join('\n')
}
btnCam.addEventListener('click', () => {
  camInfo.classList.toggle('hidden')
  btnCam.textContent = camInfo.classList.contains('hidden') ? 'camera ▾' : 'camera ▴'
  updateCamInfo()
})

// --- Recording (captures the view: split or overlay) ------------------------
let recorder = null, chunks = []
function pickMime() {
  for (const m of ['video/mp4;codecs=h264', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'])
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m
  return ''
}
function startRec() {
  const mime = pickMime()
  recorder = new MediaRecorder(viewCanvas.captureStream(30), mime ? { mimeType: mime } : undefined)
  chunks = []
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }
  recorder.onstop = () => {
    const type = recorder.mimeType || 'video/webm'
    const url = URL.createObjectURL(new Blob(chunks, { type }))
    const a = document.createElement('a'); a.href = url
    a.download = `zipdepth_${displayMode}.${type.includes('mp4') ? 'mp4' : 'webm'}`
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
  is3D = m === '3d'
  mode = is3D ? 'live' : m       // 3D sources frames from the live camera
  tabLive.classList.toggle('active', m === 'live')
  tabPhoto.classList.toggle('active', m === 'photo')
  tab3d.classList.toggle('active', is3D)
  const live = m === 'live'
  // 2D vs 3D surfaces
  viewCanvas.hidden = is3D
  cloudCanvas.hidden = !is3D
  cloudPanel.hidden = !is3D
  // 2D controls only in 2D live
  document.getElementById('view-seg').closest('.row').hidden = is3D
  btnLive.hidden = !live || is3D; btnRec.hidden = !live || is3D
  camRow.hidden = (!live && !is3D) || camSel.options.length <= 1
  zoomRow.hidden = (!live && !is3D) || !stream; btnPhotoPick.hidden = live || is3D
  if (is3D) { enter3D() }
  else if (!live) stopLive()
}
tabLive.addEventListener('click', () => switchMode('live'))
tabPhoto.addEventListener('click', () => switchMode('photo'))
tab3d.addEventListener('click', () => switchMode('3d'))
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
window.addEventListener('resize', () => { if (mode === 'photo' && photoImg) compose(photoImg, photoImg.naturalWidth, photoImg.naturalHeight) })

function showHint(t) { hint.textContent = t; hint.classList.remove('hidden') }
function hideHint() { hint.classList.add('hidden') }

// --- 3D point-cloud mode ----------------------------------------------------
function applyCloudTarget() {
  if (cam && cloud) cam.setTarget(-(cloud.params.near + cloud.params.far) / 2)
}
async function enter3D() {
  if (!cloud) {
    try {
      cloud = new CloudRenderer(cloudCanvas)
      await cloud.init()
      cam = new OrbitCamera(cloudCanvas)
      imu = new IMU()
      applyCloudTarget()
      wireCloudControls()
      cloudTick()
    } catch (e) { showHint('3D unavailable: ' + e.message + '\n(WebGPU required)'); cloud = null; return }
  }
  if (!stream) startLive()
}
function cloudTick() {
  cloudRAF = requestAnimationFrame(cloudTick)
  if (!is3D || !cloud) return
  const aspect = (cloudCanvas.clientWidth || 1) / (cloudCanvas.clientHeight || 1)
  const { view, proj } = cam.viewProj(aspect)
  cloud.render(view, proj)
}
function wireCloudControls() {
  const fov = $('c-fov'), fovV = $('c-fov-v'), sz = $('c-size'), near = $('c-near'), far = $('c-far')
  fov.addEventListener('input', () => { cloud.params.fovDeg = +fov.value; fovV.textContent = fov.value + '°' })
  sz.addEventListener('input', () => { cloud.params.radius = +sz.value })
  near.addEventListener('input', () => { cloud.params.near = +near.value; applyCloudTarget() })
  far.addEventListener('input', () => { cloud.params.far = +far.value; applyCloudTarget() })
  $('cloud-color').addEventListener('click', (e) => {
    const b = e.target.closest('.seg-btn'); if (!b) return
    cloud.params.colorMode = +b.dataset.cm
    document.querySelectorAll('#cloud-color .seg-btn').forEach((x) => x.classList.toggle('active', x === b))
  })
  $('btn-freeze').addEventListener('click', () => {
    cloudFrozen = !cloudFrozen
    $('btn-freeze').textContent = cloudFrozen ? 'Live' : 'Freeze'
    $('btn-freeze').classList.toggle('rec-on', cloudFrozen)
  })
  $('btn-recenter').addEventListener('click', () => { cam.yaw = 0; cam.pitch = 0; imu.recenter() })
  $('btn-gyro').addEventListener('click', async () => {
    if (imu.enabled) { imu.disable(); cam.setGyro(0, 0); $('btn-gyro').textContent = 'Gyro look: off'; return }
    try {
      await imu.enable((y, p) => cam.setGyro(y, p))
      $('btn-gyro').textContent = 'Gyro look: on'
    } catch (e) { showHint(e.message) }
  })
}

switchMode('live')
