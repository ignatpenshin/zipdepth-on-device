// Orbit camera for the point cloud: touch-drag to rotate, pinch to dolly.
// Also accepts an external orientation offset (yaw/pitch) from the device IMU so
// moving the phone "looks around" the frozen cloud. Produces view + proj mats.
import { perspective, lookAt } from './mat4.js'

const clamp = (v, a, b) => Math.min(b, Math.max(a, v))

export class OrbitCamera {
  constructor(el, target = [0, 0, -1.8]) {
    this.el = el
    this.target = target
    this.yaw = 0            // radians, from touch
    this.pitch = 0
    this.radius = 1.8
    this.fovY = 50 * Math.PI / 180
    this.gyaw = 0           // radians, from gyro (added on top of touch)
    this.gpitch = 0
    this._bind()
  }

  setTarget(z) { this.target = [0, 0, z]; this.radius = Math.abs(z) }

  setGyro(yaw, pitch) { this.gyaw = yaw; this.gpitch = pitch }

  _bind() {
    let lastX = 0, lastY = 0, dragging = false, pinch0 = 0, r0 = 0
    const dist = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY)
    this.el.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) { dragging = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY }
      else if (e.touches.length === 2) { dragging = false; pinch0 = dist(e.touches); r0 = this.radius }
    }, { passive: true })
    this.el.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2 && pinch0) {
        e.preventDefault()
        this.radius = clamp(r0 * (pinch0 / dist(e.touches)), 0.15, 30)
      } else if (dragging && e.touches.length === 1) {
        const t = e.touches[0]
        this.yaw -= (t.clientX - lastX) * 0.006
        this.pitch = clamp(this.pitch + (t.clientY - lastY) * 0.006, -1.4, 1.4)
        lastX = t.clientX; lastY = t.clientY
      }
    }, { passive: false })
    this.el.addEventListener('touchend', () => { dragging = false; pinch0 = 0 })
    // desktop mouse (for local dev)
    let down = false
    this.el.addEventListener('mousedown', (e) => { down = true; lastX = e.clientX; lastY = e.clientY })
    window.addEventListener('mouseup', () => { down = false })
    window.addEventListener('mousemove', (e) => {
      if (!down) return
      this.yaw -= (e.clientX - lastX) * 0.006
      this.pitch = clamp(this.pitch + (e.clientY - lastY) * 0.006, -1.4, 1.4)
      lastX = e.clientX; lastY = e.clientY
    })
    this.el.addEventListener('wheel', (e) => {
      e.preventDefault(); this.radius = clamp(this.radius * (1 + Math.sign(e.deltaY) * 0.08), 0.15, 30)
    }, { passive: false })
  }

  viewProj(aspect) {
    const yaw = this.yaw + this.gyaw, pitch = clamp(this.pitch + this.gpitch, -1.45, 1.45)
    const cp = Math.cos(pitch), sp = Math.sin(pitch)
    const eye = [
      this.target[0] + this.radius * cp * Math.sin(yaw),
      this.target[1] + this.radius * sp,
      this.target[2] + this.radius * cp * Math.cos(yaw),
    ]
    return {
      view: lookAt(eye, this.target, [0, 1, 0]),
      proj: perspective(this.fovY, aspect, 0.01, 100),
    }
  }
}
