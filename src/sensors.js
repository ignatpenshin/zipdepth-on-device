// Native-ready sensor seam.
//
// The point-cloud engine consumes "sensor frames" with this shape:
//
//   Frame = {
//     rgb:        Uint8ClampedArray(size*size*4),  // aligned color crop
//     depth:      Float32Array(size*size),         // inverse depth (web) OR metric meters (native)
//     size:       number,
//     metric:     boolean,                         // true if depth is real metric meters
//     intrinsics: { fx, fy, cx, cy } | null,       // pixels; null => assume FOV
//     pose:       Float32Array(16) | null,          // world<-camera, column-major; null => single-view
//   }
//
// WEB (this file): rgb from the camera crop, depth from ZipDepth (inverse,
// non-metric), intrinsics null (assumed FOV), pose null (IMU gives view-only
// orientation, not a metric camera pose).
//
// NATIVE (future ARKit/ARCore WKWebView shell): the shell posts frames with
// metric `depth` (LiDAR sceneDepth / ToF), real `intrinsics`, and a drift-
// corrected `pose` — same shape, so renderer + fusion need no changes.

const rad = (d) => d * Math.PI / 180

// Device orientation → view-only look-around (yaw/pitch), relative to the
// orientation at start. iOS gates this behind a user-gesture permission call.
export class IMU {
  constructor() {
    this.yaw = 0; this.pitch = 0
    this.enabled = false
    this._ref = null
    this._gain = 1.1
    this._onCb = null
    this._handler = (e) => {
      if (e.beta == null || e.gamma == null) return
      if (!this._ref) this._ref = { beta: e.beta, gamma: e.gamma }
      // gamma = left/right tilt -> yaw; beta = front/back tilt -> pitch
      this.yaw = rad((e.gamma - this._ref.gamma)) * this._gain
      this.pitch = rad((e.beta - this._ref.beta)) * this._gain * 0.8
      if (this._onCb) this._onCb(this.yaw, this.pitch)
    }
  }

  static get needsPermission() {
    return typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function'
  }

  async enable(onChange) {
    if (IMU.needsPermission) {
      const res = await DeviceOrientationEvent.requestPermission()
      if (res !== 'granted') throw new Error('Motion permission denied')
    }
    this._onCb = onChange
    this._ref = null
    window.addEventListener('deviceorientation', this._handler)
    this.enabled = true
  }

  recenter() { this._ref = null }

  disable() {
    window.removeEventListener('deviceorientation', this._handler)
    this.enabled = false; this.yaw = 0; this.pitch = 0; this._ref = null
    if (this._onCb) this._onCb(0, 0)
  }
}
