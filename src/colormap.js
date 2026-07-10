// Matplotlib "Spectral" anchor colors (low → high), matching ZipDepth's viz.
const SPECTRAL = [
  [158, 1, 66], [213, 62, 79], [244, 109, 67], [253, 174, 97],
  [254, 224, 139], [255, 255, 191], [230, 245, 152], [171, 221, 164],
  [102, 194, 165], [50, 136, 189], [94, 79, 162],
]

// Build a 256-entry RGB LUT. `invert:true` (repo default) => near (max depth
// value) maps to warm red, far maps to blue/purple.
function buildLUT(invert = true) {
  const lut = new Uint8Array(256 * 3)
  const n = SPECTRAL.length - 1
  for (let i = 0; i < 256; i++) {
    const t = (invert ? 255 - i : i) / 255
    const f = t * n
    const j = Math.min(Math.floor(f), n - 1)
    const a = f - j
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = Math.round(SPECTRAL[j][c] * (1 - a) + SPECTRAL[j + 1][c] * a)
    }
  }
  return lut
}

const LUT = buildLUT(true)

// depth: Float32Array (H*W). Writes colorized RGBA into `out` (Uint8ClampedArray,
// H*W*4). Per-frame min/max normalization, exactly like depth_to_colormap().
export function colorizeInto(depth, out, alpha = 255) {
  let min = Infinity, max = -Infinity
  for (let i = 0; i < depth.length; i++) {
    const v = depth[i]
    if (v < min) min = v
    if (v > max) max = v
  }
  const scale = 255 / (max - min + 1e-8)
  for (let i = 0; i < depth.length; i++) {
    let u = ((depth[i] - min) * scale) | 0
    if (u < 0) u = 0; else if (u > 255) u = 255
    const o = i * 4, l = u * 3
    out[o] = LUT[l]; out[o + 1] = LUT[l + 1]; out[o + 2] = LUT[l + 2]; out[o + 3] = alpha
  }
}
