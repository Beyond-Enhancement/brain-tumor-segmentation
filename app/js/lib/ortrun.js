// ONNX Runtime Web wrapper: loads a network (graph + compact weight pack), picks WebGPU or the
// multi-threaded CPU (WebAssembly) backend, and runs single patches.
let ort = null;
let ortBase = null;

export async function loadOrt(base) {
  if (ort) return ort;
  ortBase = base;
  ort = await import(base + 'vendor/ort/ort.webgpu.bundle.min.mjs');
  ort.env.wasm.wasmPaths = { wasm: base + 'vendor/ort/ort-wasm-simd-threaded.asyncify.wasm' };
  ort.env.wasm.numThreads = Math.max(1, Math.min(16, (self.navigator && navigator.hardwareConcurrency) || 4));
  ort.env.logLevel = 'error';
  return ort;
}

export async function gpuInfo() {
  try {
    if (!self.navigator || !navigator.gpu) return null;
    const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!a) return null;
    const info = a.info || {};
    return {
      vendor: info.vendor || '', architecture: info.architecture || '', description: info.description || '',
      maxBuffer: a.limits.maxBufferSize, maxStorage: a.limits.maxStorageBufferBindingSize,
      software: /swiftshader|llvmpipe|warp|basic render/i.test((info.vendor || '') + ' ' + (info.architecture || '') + ' ' + (info.description || '')),
    };
  } catch (e) { return null; }
}

// IEEE half -> float lookup table
let F16 = null;
function f16table() {
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return F16;
}

// Rebuild the fp32 external-data blob from the pack described in cfg.weights.tensors.
// Each entry: { offset (bytes in external fp32 blob), count, fmt: 'f16'|'q8'|'f32', pack (byte offset), channels }
export function unpackWeights(pack, wcfg) {
  const outBytes = new Uint8Array(wcfg.externalBytes);
  const out = new DataView(outBytes.buffer);
  const T = f16table();
  const src = new DataView(pack.buffer, pack.byteOffset, pack.byteLength);
  for (const t of wcfg.tensors) {
    if (t.fmt === 'f16') {
      for (let i = 0; i < t.count; i++) out.setFloat32(t.offset + 4 * i, T[src.getUint16(t.pack + 2 * i, true)], true);
    } else if (t.fmt === 'q8') {
      const per = t.count / t.channels;
      for (let c = 0; c < t.channels; c++) {
        const s = src.getFloat32(t.scales + 4 * c, true);
        for (let i = 0; i < per; i++) out.setFloat32(t.offset + 4 * (c * per + i), src.getInt8(t.pack + c * per + i) * s, true);
      }
    } else { // raw bytes (fp32 or non-float tensors)
      outBytes.set(pack.subarray(t.pack, t.pack + t.bytes), t.offset);
    }
  }
  return outBytes;
}

export class Network {
  constructor(cfg, session, backend) { this.cfg = cfg; this.session = session; this.backend = backend; }
  static async create(base, cfg, backend, onProgress) {
    await loadOrt(base);
    const dir = base + cfg.dir;
    const fetchBytes = async name => {
      if (cfg.inline && cfg.inline[name]) return cfg.inline[name]; // add-on model held in memory
      const r = await fetch(dir + name);
      if (!r.ok) throw new Error('Missing model file ' + name);
      return new Uint8Array(await r.arrayBuffer());
    };
    if (onProgress) onProgress('Loading ' + cfg.name + ' weights');
    const [graph, pack] = await Promise.all([fetchBytes(cfg.onnx), fetchBytes(cfg.weights.pack)]);
    const ext = unpackWeights(pack, cfg.weights);
    const opts = {
      executionProviders: backend === 'webgpu' ? [{ name: 'webgpu', preferredLayout: 'NCHW' }] : ['wasm'],
      graphOptimizationLevel: 'all',
      externalData: [{ path: cfg.weights.externalName, data: ext }],
      enableMemPattern: backend !== 'webgpu',
      logSeverityLevel: 3,
    };
    if (onProgress) onProgress('Starting ' + cfg.name + ' (' + (backend === 'webgpu' ? 'GPU' : 'CPU') + ')');
    const session = await ort.InferenceSession.create(graph, opts);
    return new Network(cfg, session, backend);
  }
  async run(patch, dims) {
    const t = new ort.Tensor('float32', patch, dims);
    const out = await this.session.run({ [this.session.inputNames[0]]: t });
    const o = out[this.session.outputNames[0]];
    const d = o.location === 'gpu-buffer' ? await o.getData(true) : o.data;
    return d;
  }
  async release() { try { await this.session.release(); } catch (e) { /* ignore */ } }
}
