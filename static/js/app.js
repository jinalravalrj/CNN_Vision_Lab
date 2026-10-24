/* ─────────────────────────────────────────────────────────────────────────
   CNN Vision Lab — app.js
   Single-page application logic for all seven pages.
   ───────────────────────────────────────────────────────────────────────── */

"use strict";

// ── Utility helpers ──────────────────────────────────────────────────────────

function qs(sel, root = document) { return root.querySelector(sel); }
function qsa(sel, root = document) { return [...root.querySelectorAll(sel)]; }
function el(tag, cls = "", html = "") {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}
function show(elem) { if (elem) elem.hidden = false; }
function hide(elem) { if (elem) elem.hidden = true; }

function toast(msg, type = "info", durationMs = 4000) {
  const container = qs("#toast-container");
  const icons = { success: "✅", error: "❌", info: "ℹ️", warning: "⚠️" };
  const t = el("div", `toast ${type}`, `<span>${icons[type] || "ℹ️"}</span><span>${msg}</span>`);
  container.appendChild(t);
  setTimeout(() => t.remove(), durationMs);
}

function showLoading(text = "Processing…") {
  const overlay = qs("#loading-overlay");
  if (!overlay) return;
  const txt = qs("#loading-text");
  if (txt) txt.textContent = text;
  overlay.hidden = false;
  overlay.classList.remove("hidden");
  overlay.style.display = "flex";
}

function hideLoading() {
  const overlay = qs("#loading-overlay");
  if (!overlay) return;
  overlay.hidden = true;
  overlay.classList.add("hidden");
  overlay.style.display = "none";
}

function fmtDate(isoStr) {
  try {
    return new Date(isoStr).toLocaleString("en-IN", {
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  } catch { return isoStr; }
}

function fmtNum(n) {
  if (n == null) return "—";
  return Number(n).toLocaleString();
}

// ── Navigation ───────────────────────────────────────────────────────────────

const PAGES = ["dashboard", "classify", "architecture", "learn", "performance", "history", "about"];
let currentPage = "dashboard";

function navigateTo(page) {
  if (!PAGES.includes(page)) return;
  currentPage = page;

  // Deactivate all pages / nav items
  qsa(".page").forEach(p => p.classList.remove("active"));
  qsa(".nav-item").forEach(n => n.classList.remove("active"));

  // Activate target
  const pageEl = qs(`#page-${page}`);
  const navEl  = qs(`#nav-${page}`);
  if (pageEl) pageEl.classList.add("active");
  if (navEl)  navEl.classList.add("active");

  // Close mobile sidebar
  qs("#sidebar").classList.remove("open");
  qs("#overlay").classList.remove("open");

  // Page-specific loaders
  if (page === "dashboard")     loadDashboard();
  if (page === "history")       loadHistory();
  if (page === "performance")   loadPerformance();
  if (page === "architecture")  initArchitecture();

  window.scrollTo(0, 0);
}

// Nav link click handlers
qsa(".nav-item").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    navigateTo(link.dataset.page);
  });
});

// Hamburger + overlay
qs("#hamburger").addEventListener("click", () => {
  qs("#sidebar").classList.toggle("open");
  qs("#overlay").classList.toggle("open");
});
qs("#overlay").addEventListener("click", () => {
  qs("#sidebar").classList.remove("open");
  qs("#overlay").classList.remove("open");
});

// ── Health check ─────────────────────────────────────────────────────────────

async function checkHealth() {
  const dot   = qs("#status-dot");
  const label = qs("#status-label");
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      dot.className = "status-dot ok";
      label.textContent = "Model ready";
    } else {
      dot.className = "status-dot error";
      label.textContent = data.model_error ? "Model error" : "Model loading…";
    }
  } catch {
    dot.className = "status-dot error";
    label.textContent = "Server offline";
  }
}

checkHealth();
setInterval(checkHealth, 30000); // re-check every 30 s

// ── Dashboard ─────────────────────────────────────────────────────────────────

async function loadDashboard() {
  // Model name
  try {
    const res  = await fetch("/api/model-info");
    const info = await res.json();
    qs("#dash-model-name").textContent = info.model_name || "—";
  } catch { /* ignore */ }

  // History count + recent list
  try {
    const res  = await fetch("/api/history?limit=5");
    const data = await res.json();
    if (data.success) {
      qs("#dash-history-count").textContent = fmtNum(data.total);
      const list = qs("#dash-recent-list");
      if (data.records.length === 0) {
        list.innerHTML = '<p class="empty-state">No predictions yet. Try the classifier!</p>';
      } else {
        list.innerHTML = "";
        data.records.forEach(r => {
          const item = el("div", "recent-item",
            `<span class="recent-label">${r.predicted}</span>
             <span class="recent-conf">${r.confidence}%</span>
             <span class="recent-ts">${fmtDate(r.timestamp)}</span>`
          );
          list.appendChild(item);
        });
      }
    }
  } catch { /* ignore */ }
}

// ── Image Classification ──────────────────────────────────────────────────────

let classifyFile = null;    // current File object
let featureFile  = null;    // copy for feature maps

// Sample images (static files shipped with the project)
const SAMPLES = [
  { name: "Cat",         file: "cat.jpg" },
  { name: "Dog",         file: "dog.jpg" },
  { name: "Car",         file: "car.jpg" },
  { name: "Bird",        file: "bird.jpg" },
  { name: "Flower",      file: "flower.jpg" },
  { name: "Mountain",    file: "mountain.jpg" },
];

function initSamples() {
  const grid = qs("#sample-grid");
  grid.innerHTML = "";
  SAMPLES.forEach(s => {
    const wrap = el("div", "sample-thumb");
    const img  = document.createElement("img");
    img.src = `/static/images/samples/${s.file}`;
    img.alt = s.name;
    img.title = s.name;
    img.onerror = () => { wrap.style.display = "none"; };
    img.onload  = () => {};
    wrap.appendChild(img);
    wrap.addEventListener("click", () => loadSampleImage(s.file, s.name));
    grid.appendChild(wrap);
  });
}

async function loadSampleImage(filename, label) {
  try {
    const res  = await fetch(`/static/images/samples/${filename}`);
    const blob = await res.blob();
    const file = new File([blob], filename, { type: blob.type });
    setClassifyFile(file);
  } catch {
    toast(`Could not load sample image "${label}".`, "error");
  }
}

function setClassifyFile(file) {
  classifyFile = file;
  featureFile  = file;
  const reader = new FileReader();
  reader.onload = e => {
    qs("#img-preview").src = e.target.result;
    show(qs("#preview-wrapper"));
    qs("#dropzone-content").hidden = true;
  };
  reader.readAsDataURL(file);
  // Clear previous results
  qs("#results-area").innerHTML = `
    <div class="results-empty">
      <div class="results-empty-icon">🔍</div>
      <p>Click <strong>Classify</strong> to analyse this image</p>
    </div>`;
  qs("#feature-maps-card").style.display = "none";
}

// Dropzone setup
(function() {
  const dz        = qs("#dropzone");
  const fileInput = qs("#file-input");
  const browseBtn = qs("#browse-btn");
  const changeBtn = qs("#change-img-btn");
  const predictBtn = qs("#predict-btn");

  browseBtn.addEventListener("click", () => fileInput.click());
  changeBtn.addEventListener("click", () => {
    classifyFile = null;
    fileInput.value = "";
    qs("#dropzone-content").hidden = false;
    hide(qs("#preview-wrapper"));
    qs("#results-area").innerHTML = `
      <div class="results-empty">
        <div class="results-empty-icon">🔍</div>
        <p>Upload an image and click <strong>Classify</strong> to see predictions</p>
      </div>`;
    qs("#feature-maps-card").style.display = "none";
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setClassifyFile(fileInput.files[0]);
  });

  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", e => {
    e.preventDefault();
    dz.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["jpg", "jpeg", "png"].includes(ext)) {
      toast("Only JPG, JPEG, and PNG files are accepted.", "error");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast("File must be smaller than 10 MB.", "error");
      return;
    }
    setClassifyFile(file);
  });

  predictBtn.addEventListener("click", runClassify);
})();

async function runClassify() {
  if (!classifyFile) { toast("Please select an image first.", "warning"); return; }

  const ext = classifyFile.name.split(".").pop().toLowerCase();
  if (!["jpg", "jpeg", "png"].includes(ext)) {
    toast("Only JPG, JPEG, and PNG files are accepted.", "error");
    return;
  }
  if (classifyFile.size > 10 * 1024 * 1024) {
    toast("File must be smaller than 10 MB.", "error");
    return;
  }

  showLoading("Classifying image…");

  const fd = new FormData();
  fd.append("image", classifyFile);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000); // 20 s timeout

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: fd,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let errMsg = `Server returned status ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.error) errMsg = errJson.error;
      } catch { /* non-JSON response */ }
      renderClassifyError(errMsg);
      toast(errMsg, "error");
      return;
    }

    const data = await res.json();
    if (!data.success) {
      const errMsg = data.error || "Prediction failed.";
      renderClassifyError(errMsg);
      toast(errMsg, "error");
      return;
    }

    renderResults(data);
    loadFeatureMaps(featureFile);

    // Refresh dashboard count
    loadDashboard();
  } catch (err) {
    clearTimeout(timeoutId);
    let errMsg = "Network error: could not reach the server.";
    if (err.name === "AbortError") {
      errMsg = "Classification request timed out after 20 seconds. Please check server load and try again.";
    }
    renderClassifyError(errMsg);
    toast(errMsg, "error");
  } finally {
    hideLoading();
  }
}

function renderClassifyError(errorMessage) {
  const area = qs("#results-area");
  if (!area) return;
  area.innerHTML = `
    <div class="results-empty" style="padding:1.5rem 1rem">
      <div class="results-empty-icon" style="color:var(--danger, #f85149)">⚠️</div>
      <p style="color:var(--danger, #f85149);font-weight:600;margin-bottom:0.5rem">Classification Failed</p>
      <p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:1rem;max-width:280px;margin-left:auto;margin-right:auto">${errorMessage}</p>
      <div style="display:flex;gap:0.5rem;justify-content:center">
        <button class="btn btn-primary btn-sm" id="retry-classify-btn">🔄 Retry</button>
        <button class="btn btn-outline btn-sm" onclick="qs('#change-img-btn').click()">Change Image</button>
      </div>
    </div>`;
  const retryBtn = qs("#retry-classify-btn");
  if (retryBtn) retryBtn.addEventListener("click", runClassify);
}

function renderResults(data) {
  const area = qs("#results-area");
  const top3 = data.top3 || [];

  const barsHtml = top3.map((item, i) => `
    <div class="top3-item">
      <div class="top3-label-row">
        <span>${i + 1}. ${item.label.replace(/_/g, " ")}</span>
        <strong>${item.confidence}%</strong>
      </div>
      <div class="top3-bar-track">
        <div class="top3-bar-fill" style="width:${item.confidence}%"></div>
      </div>
    </div>`).join("");

  area.innerHTML = `
    <div class="result-main">
      <div class="result-label">${data.predicted}</div>
      <div class="result-conf">${data.confidence}%</div>
      <div class="result-meta">
        Model: ${data.model_name} &nbsp;|&nbsp; Inference: ${data.infer_ms} ms
      </div>
    </div>
    <div style="margin-bottom:1rem">
      <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.75rem;font-weight:600">TOP 3 PREDICTIONS</div>
      <div class="top3-list">${barsHtml}</div>
    </div>
    <div class="result-actions">
      <button class="btn btn-outline btn-sm" onclick="qs('#change-img-btn').click()">Try Another</button>
      <button class="btn btn-outline btn-sm" onclick="navigateTo('history')">View History</button>
    </div>
    <p style="font-size:0.72rem;color:var(--text-dim);margin-top:1rem;text-align:center">
      Prediction saved to SQLite history at ${fmtDate(data.timestamp)}
    </p>`;
}

async function loadFeatureMaps(file) {
  const card = qs("#feature-maps-card");
  if (!file) return;

  const fd = new FormData();
  fd.append("image", file);

  try {
    const res  = await fetch("/api/feature-maps", { method: "POST", body: fd });
    const data = await res.json();
    if (!data.success) return;

    const grid = qs("#feature-maps-grid");
    grid.innerHTML = "";
    data.maps.forEach((b64, i) => {
      const img = document.createElement("img");
      img.src   = `data:image/png;base64,${b64}`;
      img.alt   = `Filter ${i + 1}`;
      img.title = `Filter ${i + 1} — Conv1 layer (shape ${data.shape.join("×")})`;
      grid.appendChild(img);
    });

    card.style.display = "block";
  } catch { /* silently skip */ }
}

// Initialise classify page once
initSamples();

// ── Architecture Diagram ──────────────────────────────────────────────────────

const ARCH_LAYERS = [
  {
    icon: "🖼️", name: "Input Image", type: "Data",
    desc: "Raw pixel data fed into the network. MobileNetV2 expects a 224×224 RGB image with pixel values in [−1, 1] after preprocessing.",
    shape_in: "224 × 224 × 3", shape_out: "224 × 224 × 3",
    params: "0 (no learnable weights)",
    example: "A 224×224 cat photo → tensor of shape (1, 224, 224, 3).",
  },
  {
    icon: "⚙️", name: "Convolution", type: "Conv2D",
    desc: "A bank of learnable filters slides (convolves) across the image spatially. Each filter produces a 2-D feature map that responds strongly wherever the corresponding pattern is detected.",
    shape_in: "H × W × C_in", shape_out: "H′ × W′ × C_out",
    params: "Kernel (k×k×C_in×C_out) + bias (C_out).",
    example: "A 3×3 filter looking for vertical edges: output high where a vertical edge appears.",
  },
  {
    icon: "⚡", name: "ReLU Activation", type: "Activation",
    desc: "Rectified Linear Unit: replaces every negative activation with 0, keeping positives unchanged. Introduces non-linearity, allowing the network to learn complex functions.",
    shape_in: "H × W × C", shape_out: "H × W × C (same shape)",
    params: "0 (no learnable weights)",
    example: "f(x) = max(0, x). An activation of −2.3 becomes 0; +1.7 stays 1.7.",
  },
  {
    icon: "⬇️", name: "Max Pooling", type: "MaxPool2D",
    desc: "Downsamples spatial dimensions by keeping the maximum value in each pool window. Provides spatial invariance and reduces computation in deeper layers.",
    shape_in: "H × W × C", shape_out: "H/2 × W/2 × C",
    params: "0 (no learnable weights)",
    example: "2×2 pool stride 2: a 112×112 map becomes 56×56.",
  },
  {
    icon: "⚙️", name: "Convolution 2", type: "Conv2D",
    desc: "A second (deeper) convolutional block. Deeper filters learn more abstract patterns — corners, textures, object parts — built on top of the low-level features from earlier layers.",
    shape_in: "H′ × W′ × C_in", shape_out: "H″ × W″ × C_out",
    params: "Kernel (k×k×C_in×C_out) + bias.",
    example: "A 3×3 filter at this depth might detect fur textures rather than single edges.",
  },
  {
    icon: "⚡", name: "ReLU 2", type: "Activation",
    desc: "Second ReLU — same purpose as the first: zero out negative activations to maintain non-linearity after each convolution.",
    shape_in: "H × W × C", shape_out: "H × W × C",
    params: "0",
    example: "Applied after every Conv2D to prevent the network from collapsing to a linear function.",
  },
  {
    icon: "⬇️", name: "Max Pooling 2", type: "MaxPool2D",
    desc: "Second pooling block. Progressively reduces spatial size so that deeper layers can see larger receptive fields without increasing filter size.",
    shape_in: "H × W × C", shape_out: "H/2 × W/2 × C",
    params: "0",
    example: "After two pool layers with stride 2, a 224×224 input shrinks to 56×56.",
  },
  {
    icon: "🔗", name: "Global Avg Pool / Flatten", type: "GAP / Flatten",
    desc: "Global Average Pooling averages each feature map to a single value, converting (H×W×C) → (C,). Alternatively, Flatten unrolls all values to a 1-D vector. Both bridge feature maps and dense layers.",
    shape_in: "H × W × C", shape_out: "C (GAP) or H×W×C (Flatten)",
    params: "0",
    example: "A 7×7×512 feature map → 512-element vector via GAP.",
  },
  {
    icon: "🧮", name: "Dense (FC)", type: "Dense",
    desc: "Fully-connected layer: every input neuron connects to every output neuron. Learns high-level combinations of extracted features for the final classification decision.",
    shape_in: "N (flat vector)", shape_out: "M (hidden units)",
    params: "N × M + M (weights + biases)",
    example: "512 → 256 Dense layer: 512×256 + 256 = 131,328 parameters.",
  },
  {
    icon: "🏷️", name: "Output / Softmax", type: "Dense + Softmax",
    desc: "Final Dense layer with a Softmax activation. Converts raw scores (logits) into a probability distribution over all classes. The class with the highest probability is the prediction.",
    shape_in: "M", shape_out: "K (number of classes)",
    params: "M × K + K",
    example: "ImageNet: 1,000 outputs. Softmax ensures all 1,000 values sum to 1.0.",
  },
];

let selectedLayer = -1;

function initArchitecture() {
  const diagram = qs("#arch-diagram");
  if (diagram.childElementCount > 0) return; // already built

  ARCH_LAYERS.forEach((layer, i) => {
    const wrap = el("div", "arch-layer-wrap");

    const div = el("div", "arch-layer",
      `<div class="arch-layer-icon">${layer.icon}</div>
       <div class="arch-layer-name">${layer.name}</div>
       <div class="arch-layer-type">${layer.type}</div>`
    );
    div.addEventListener("click", () => showLayerDetail(i, div));
    wrap.appendChild(div);

    if (i < ARCH_LAYERS.length - 1) {
      wrap.appendChild(el("div", "arch-arrow", "→"));
    }
    diagram.appendChild(wrap);
  });

  // Load MV2 layer table
  loadMV2Layers();
}

function showLayerDetail(index, clickedEl) {
  qsa(".arch-layer").forEach(l => l.classList.remove("selected"));
  clickedEl.classList.add("selected");
  selectedLayer = index;

  const layer  = ARCH_LAYERS[index];
  const panel  = qs("#layer-detail-panel");
  panel.innerHTML = `
    <div class="layer-detail-name">${layer.icon} ${layer.name}</div>
    <div class="layer-detail-desc">${layer.desc}</div>
    <div class="layer-detail-grid">
      <div>
        <div class="layer-kv"><strong>Type:</strong> ${layer.type}</div>
        <div class="layer-kv"><strong>Input shape:</strong> ${layer.shape_in}</div>
        <div class="layer-kv"><strong>Output shape:</strong> ${layer.shape_out}</div>
        <div class="layer-kv"><strong>Parameters:</strong> ${layer.params}</div>
      </div>
      <div>
        <div class="layer-kv"><strong>Example:</strong> ${layer.example}</div>
      </div>
    </div>`;
}

async function loadMV2Layers() {
  try {
    const res  = await fetch("/api/model-info");
    const info = await res.json();

    const tbody = qs("#mv2-table-body");
    tbody.innerHTML = `
      <tr><td>—</td><td>Conv1</td><td>Conv2D (stride 2)</td><td>112 × 112 × 32</td><td>864</td></tr>
      <tr><td>—</td><td>bn_Conv1</td><td>BatchNorm</td><td>112 × 112 × 32</td><td>128</td></tr>
      <tr><td>—</td><td>expanded_conv_depthwise</td><td>DepthwiseConv2D</td><td>112 × 112 × 32</td><td>288</td></tr>
      <tr><td>—</td><td>expanded_conv_project</td><td>Conv2D (1×1)</td><td>112 × 112 × 16</td><td>512</td></tr>
      <tr><td>…</td><td>(16 inverted residual blocks)</td><td>InvertedResidual</td><td>various</td><td>~3.4 M total</td></tr>
      <tr><td>—</td><td>Conv_1</td><td>Conv2D (1×1)</td><td>7 × 7 × 1280</td><td>409,600</td></tr>
      <tr><td>—</td><td>global_average_pooling2d</td><td>GlobalAvgPool</td><td>1280</td><td>0</td></tr>
      <tr><td>—</td><td>Logits</td><td>Dense + Softmax</td><td>1,000</td><td>1,281,000</td></tr>`;

    if (info.total_params) {
      const row = el("tr", "", `<td colspan="4" style="color:var(--teal);font-weight:600">Total (loaded model)</td><td>${fmtNum(info.total_params)}</td>`);
      tbody.appendChild(row);
    }
  } catch { /* ignore */ }
}

// ── Learn CNN ────────────────────────────────────────────────────────────────

const MODULES = [
  {
    title: "1. Neural Network Basics",
    content: `
      <p>A <strong>neural network</strong> is a computational model loosely inspired by the biological brain. It consists of layers of interconnected units called <em>neurons</em>.</p>
      <h4>Key components</h4>
      <ul>
        <li><strong>Neuron:</strong> Receives inputs, applies weights, adds a bias, passes through an activation function.</li>
        <li><strong>Layer:</strong> A group of neurons operating in parallel at the same depth.</li>
        <li><strong>Weight (w):</strong> A learnable parameter that scales the input signal.</li>
        <li><strong>Bias (b):</strong> A learnable offset added after weighting.</li>
      </ul>
      <h4>Simple neuron equation</h4>
      <div class="code-block">output = activation( w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b )</div>
      <p>During <em>training</em>, the network adjusts w and b to minimise prediction error on labelled examples.</p>`,
  },
  {
    title: "2. CNN Fundamentals",
    content: `
      <p>A <strong>Convolutional Neural Network (CNN)</strong> is specialised for processing grid-structured data like images. Unlike fully-connected networks, CNNs exploit spatial locality and parameter sharing.</p>
      <h4>Why CNNs for images?</h4>
      <ul>
        <li>A 224×224 RGB image has 150,528 pixels. A dense layer would need millions of parameters just for the first layer.</li>
        <li>CNNs use small <em>local filters</em> (e.g., 3×3) that slide across the image — drastically fewer parameters.</li>
        <li>The same filter is reused at every position (weight sharing), enabling translation invariance.</li>
      </ul>
      <h4>Typical CNN structure</h4>
      <div class="code-block">Input → [Conv → ReLU → Pool] × N → Flatten/GAP → Dense → Softmax</div>`,
  },
  {
    title: "3. Convolution & Kernels",
    content: `
      <p>The <strong>convolution operation</strong> slides a small matrix called a <strong>kernel (filter)</strong> across the input image and computes a dot product at each position.</p>
      <h4>Example: 3×3 edge-detection kernel</h4>
      <div class="code-block">[-1, -1, -1]
[ 0,  0,  0]
[ 1,  1,  1]</div>
      <p>When this kernel is convolved with an image, it produces a strong positive response at horizontal edges (dark-above, light-below) and a strong negative response at the reverse.</p>
      <h4>Feature map</h4>
      <p>The result of applying one kernel across the full input is called a <em>feature map</em>. With C filters, you get C feature maps — forming the next layer's depth dimension.</p>
      <h4>Output size formula</h4>
      <div class="code-block">out = floor((W - K + 2P) / S) + 1
W = input width, K = kernel size, P = padding, S = stride</div>`,
  },
  {
    title: "4. Stride & Padding",
    content: `
      <h4>Stride (S)</h4>
      <p>The number of pixels the filter moves at each step. Stride 1 → dense coverage; Stride 2 → halves the output size (like pooling).</p>
      <h4>Padding (P)</h4>
      <p>Zeros added around the input border to control output size:</p>
      <ul>
        <li><strong>Valid padding (P=0):</strong> No padding; output is smaller than input.</li>
        <li><strong>Same padding:</strong> Pad so output has the same spatial size as input (with stride 1).</li>
      </ul>
      <h4>Example</h4>
      <div class="code-block">Input: 28×28, Kernel: 3×3, Stride: 1, Padding: 0
→ Output: (28-3+0)/1 + 1 = 26×26

Input: 28×28, Kernel: 3×3, Stride: 1, Padding: 1 (same)
→ Output: 28×28</div>`,
  },
  {
    title: "5. ReLU Activation",
    content: `
      <p><strong>ReLU</strong> (Rectified Linear Unit) is the most widely used activation function in CNNs.</p>
      <h4>Formula</h4>
      <div class="code-block">f(x) = max(0, x)</div>
      <p>Negative values become 0; positive values pass through unchanged.</p>
      <h4>Why ReLU?</h4>
      <ul>
        <li><strong>Non-linearity:</strong> Without activation functions, stacking linear layers is still linear.</li>
        <li><strong>Sparsity:</strong> Negative activations become exactly 0, creating sparse representations.</li>
        <li><strong>No vanishing gradient:</strong> For positive inputs, the gradient is exactly 1 — no saturation.</li>
        <li><strong>Computation:</strong> Extremely cheap — just a comparison with zero.</li>
      </ul>
      <h4>Variants</h4>
      <ul>
        <li><strong>Leaky ReLU:</strong> f(x) = x if x > 0 else 0.01·x</li>
        <li><strong>ELU, GELU, Swish</strong> — used in more modern architectures.</li>
      </ul>`,
  },
  {
    title: "6. Max Pooling",
    content: `
      <p><strong>Max Pooling</strong> downsamples the spatial dimensions of feature maps by selecting the maximum value in each pool window.</p>
      <h4>Example: 2×2 Max Pool, Stride 2</h4>
      <div class="code-block">Input 4×4:          After pool 2×2:
1  3  2  4          6  8
5  6  1  2   →      3  9
0  2  3  1
4  1  8  9</div>
      <h4>Benefits</h4>
      <ul>
        <li>Reduces computation in subsequent layers.</li>
        <li>Provides a degree of spatial invariance (small shifts don't change the max).</li>
        <li>Retains the strongest activations (dominant features).</li>
      </ul>
      <p><strong>Average Pooling</strong> is an alternative that averages instead of taking the max. MobileNetV2 uses <em>Global Average Pooling</em> at the end.</p>`,
  },
  {
    title: "7. Flattening & Dense Layers",
    content: `
      <p>After the convolutional blocks, we need to convert the 3-D feature maps to a 1-D vector before the classification layers.</p>
      <h4>Flatten</h4>
      <div class="code-block">7 × 7 × 512  →  25,088 values  (7×7×512)</div>
      <p>Unrolls all dimensions into a single long vector. This is then fed to a Dense layer.</p>
      <h4>Global Average Pooling (GAP)</h4>
      <div class="code-block">7 × 7 × 512  →  512 values  (average of each 7×7 map)</div>
      <p>Much smaller output — dramatically reduces parameters in the next layer. Used in modern architectures (MobileNet, ResNet, EfficientNet).</p>
      <h4>Dense (Fully-Connected) Layer</h4>
      <p>Every input neuron connects to every output neuron. Learns global combinations of features:</p>
      <div class="code-block">output = ReLU( W · x + b )
W shape: (output_units, input_units)</div>`,
  },
  {
    title: "8. Softmax & Class Probabilities",
    content: `
      <p><strong>Softmax</strong> is applied to the final layer's raw scores (logits) to produce a valid probability distribution.</p>
      <h4>Formula</h4>
      <div class="code-block">softmax(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)   for j = 1…K</div>
      <ul>
        <li>All outputs are in (0, 1).</li>
        <li>All outputs sum to exactly 1.</li>
        <li>The largest logit gets the highest probability.</li>
      </ul>
      <h4>Example (3 classes)</h4>
      <div class="code-block">Logits:       [2.0,  1.0,  0.1]
Exp:          [7.39, 2.72, 1.11]  → sum = 11.22
Softmax:      [0.66, 0.24, 0.10]
Prediction:   Class 0  (66% confidence)</div>`,
  },
  {
    title: "9. Forward Propagation",
    content: `
      <p><strong>Forward propagation</strong> is the process of passing an input through the network layer by layer to produce an output.</p>
      <h4>Steps for an image</h4>
      <ol style="margin-bottom:0.75rem">
        <li>Preprocess: normalise pixel values, resize to model input size.</li>
        <li>Pass through Conv → ReLU → Pool blocks.</li>
        <li>Flatten / Global Average Pool.</li>
        <li>Pass through Dense layers.</li>
        <li>Apply Softmax to get class probabilities.</li>
        <li>Select the class with the highest probability.</li>
      </ol>
      <p>No weights are updated during forward propagation. It is a read-only pass.</p>
      <h4>Prediction vs Training</h4>
      <ul>
        <li><strong>Inference (prediction):</strong> Only forward pass. Used in production.</li>
        <li><strong>Training:</strong> Forward pass → compute loss → backward pass → update weights.</li>
      </ul>`,
  },
  {
    title: "10. Loss, Backpropagation & Gradient Descent",
    content: `
      <h4>Loss Function</h4>
      <p>Measures how wrong the network's prediction is. For multi-class classification:</p>
      <div class="code-block">Cross-Entropy Loss = −Σ yᵢ · log(ŷᵢ)
yᵢ  = true label (one-hot)
ŷᵢ  = predicted probability</div>
      <h4>Backpropagation</h4>
      <p>Computes the gradient of the loss with respect to every weight using the chain rule:</p>
      <div class="code-block">∂L/∂w = ∂L/∂output · ∂output/∂w   (chain rule)</div>
      <p>Gradients flow backward from the output layer to the input layer, layer by layer.</p>
      <h4>Gradient Descent</h4>
      <p>Updates each weight in the direction that reduces the loss:</p>
      <div class="code-block">w ← w − η · ∂L/∂w
η = learning rate (e.g., 0.001)</div>
      <ul>
        <li><strong>SGD:</strong> One sample per update.</li>
        <li><strong>Mini-batch SGD:</strong> Small batch (32–256) per update.</li>
        <li><strong>Adam:</strong> Adaptive learning rates per parameter. Most popular.</li>
      </ul>`,
  },
];

const QUIZ = [
  { q: "What is the main purpose of a convolutional filter?",
    opts: ["To randomly change pixel values", "To detect spatial patterns in the input", "To compress the image file size", "To apply colour correction"],
    ans: 1, exp: "A convolutional filter (kernel) slides across the input and detects spatial patterns such as edges, textures, and shapes." },
  { q: "What does ReLU do to negative activations?",
    opts: ["Doubles them", "Keeps them as-is", "Sets them to zero", "Inverts their sign"],
    ans: 2, exp: "ReLU = max(0, x): negative values become 0, positive values pass through unchanged." },
  { q: "What is the output size of a Conv2D with input 32×32, kernel 3×3, stride 1, no padding?",
    opts: ["32×32", "30×30", "29×29", "34×34"],
    ans: 1, exp: "Formula: (32 − 3 + 0) / 1 + 1 = 30." },
  { q: "What does Max Pooling do?",
    opts: ["Adds new features by interpolation", "Keeps the maximum value in each pool window", "Averages all values equally", "Applies learnable weights to each region"],
    ans: 1, exp: "Max Pooling keeps the largest value in each pooling window, downsampling the feature map." },
  { q: "Why is Softmax used in the output layer for classification?",
    opts: ["It makes all outputs negative", "It converts logits to a probability distribution summing to 1", "It speeds up training", "It removes noise from predictions"],
    ans: 1, exp: "Softmax converts raw scores (logits) into probabilities that sum to 1.0, allowing the model to express confidence." },
  { q: "What is weight sharing in CNNs?",
    opts: ["Two networks share the same architecture", "The same filter weights are applied at every spatial position", "Weights are shared between different datasets", "Bias values are reused across layers"],
    ans: 1, exp: "In convolution, the same filter is reused across all spatial positions. This drastically reduces the number of parameters." },
  { q: "What is Global Average Pooling?",
    opts: ["A pooling layer that averages the entire feature map to one value per channel", "A pooling layer that uses the global maximum", "A layer that flattens feature maps", "A pooling layer with a large kernel"],
    ans: 0, exp: "GAP averages each H×W feature map to a single scalar, converting (H, W, C) → (C,)." },
  { q: "Which optimiser adapts the learning rate for each parameter individually?",
    opts: ["SGD", "RMSProp", "Adam", "Adagrad"],
    ans: 2, exp: "Adam (Adaptive Moment Estimation) maintains per-parameter learning rates using first and second moment estimates." },
  { q: "What does 'backpropagation' compute?",
    opts: ["The forward pass output", "Gradients of the loss with respect to all weights", "The class probabilities", "The confusion matrix"],
    ans: 1, exp: "Backpropagation applies the chain rule to compute ∂L/∂w for every weight, enabling gradient descent." },
  { q: "MobileNetV2 uses which key building block instead of standard convolution?",
    opts: ["Dilated convolution", "Depthwise separable convolution + inverted residual blocks", "Attention mechanism", "Recurrent convolution"],
    ans: 1, exp: "MobileNetV2 uses depthwise separable convolutions inside inverted residual blocks, reducing computation ~8-9× versus standard conv." },
];

let currentModule = 0;
let quizIndex = 0;
let quizScore = 0;
let quizAnswered = false;

function buildLearnPage() {
  const nav = qs("#learn-nav");
  nav.innerHTML = "";
  MODULES.forEach((m, i) => {
    const li = el("li");
    const a  = el("a", "", m.title);
    a.addEventListener("click", e => { e.preventDefault(); showModule(i); });
    li.appendChild(a);
    nav.appendChild(li);
  });
  showModule(0);
  buildQuiz(0);
}

function showModule(index) {
  currentModule = index;
  const m = MODULES[index];
  const content = qs("#learn-content");

  // Update sidebar active
  qsa(".learn-nav li a").forEach((a, i) => {
    a.className = i === index ? "active" : "";
  });

  content.innerHTML = `
    <h3 class="module-title">${m.title}</h3>
    <div class="module-body">${m.content}</div>
    <div class="module-nav-row">
      <button class="btn btn-outline btn-sm" ${index === 0 ? "disabled" : ""} onclick="showModule(${index - 1})">← Previous</button>
      <span style="font-size:0.78rem;color:var(--text-muted)">${index + 1} / ${MODULES.length}</span>
      <button class="btn btn-outline btn-sm" ${index === MODULES.length - 1 ? "disabled" : ""} onclick="showModule(${index + 1})">Next →</button>
    </div>`;
}

function buildQuiz(index) {
  if (index >= QUIZ.length) {
    renderQuizComplete();
    return;
  }
  quizIndex = index;
  quizAnswered = false;
  const q = QUIZ[index];
  const container = qs("#quiz-container");
  const optHtml = q.opts.map((opt, i) => `
    <button class="quiz-option" onclick="answerQuiz(${i}, this)">
      <span>${String.fromCharCode(65 + i)}.</span><span>${opt}</span>
    </button>`).join("");

  container.innerHTML = `
    <div class="quiz-progress">${index + 1} / ${QUIZ.length}</div>
    <div class="quiz-question">${q.q}</div>
    <div class="quiz-options">${optHtml}</div>
    <div class="quiz-feedback" id="quiz-feedback"></div>
    <div style="display:flex;justify-content:flex-end;margin-top:0.5rem">
      <button class="btn btn-outline btn-sm" id="quiz-next-btn" style="display:none" onclick="buildQuiz(${index + 1})">Next Question →</button>
    </div>`;
}

function answerQuiz(chosenIdx, btn) {
  if (quizAnswered) return;
  quizAnswered = true;
  const q = QUIZ[quizIndex];

  qsa(".quiz-option").forEach((opt, i) => {
    if (i === q.ans)   opt.classList.add("correct");
    if (i === chosenIdx && chosenIdx !== q.ans) opt.classList.add("wrong");
  });

  if (chosenIdx === q.ans) quizScore++;

  qs("#quiz-feedback").textContent = `${chosenIdx === q.ans ? "✅ Correct!" : "❌ Incorrect."} ${q.exp}`;
  const nextBtn = qs("#quiz-next-btn");
  if (nextBtn) nextBtn.style.display = "inline-flex";
}

function renderQuizComplete() {
  const pct = Math.round((quizScore / QUIZ.length) * 100);
  const grade = pct >= 80 ? "Excellent 🎉" : pct >= 60 ? "Good 👍" : "Keep practising 📖";
  qs("#quiz-container").innerHTML = `
    <div style="text-align:center;padding:2rem">
      <div class="quiz-score">${quizScore} / ${QUIZ.length} — ${pct}%</div>
      <div style="font-size:1.1rem;margin-bottom:1.5rem">${grade}</div>
      <button class="btn btn-primary" onclick="restartQuiz()">Retake Quiz</button>
    </div>`;
}

function restartQuiz() {
  quizScore = 0;
  buildQuiz(0);
}

buildLearnPage();

// ── Prediction History ────────────────────────────────────────────────────────

let histPage = 0;
const HIST_LIMIT = 20;

async function loadHistory() {
  const offset = histPage * HIST_LIMIT;
  try {
    const res  = await fetch(`/api/history?limit=${HIST_LIMIT}&offset=${offset}`);
    const data = await res.json();
    if (!data.success) { toast("Could not load history.", "error"); return; }

    qs("#history-total").textContent = data.total;
    const container = qs("#history-container");

    if (data.records.length === 0) {
      container.innerHTML = '<p class="empty-state">No predictions recorded yet.</p>';
      qs("#pagination-row").style.display = "none";
      return;
    }

    const listEl = el("div", "history-card-list");
    data.records.forEach(r => {
      const row = el("div", "hist-row");
      row.innerHTML = `
        <span class="hist-label">${r.predicted}</span>
        <span class="hist-file" title="${r.filename}">${r.filename}</span>
        <span class="hist-conf">${r.confidence}%</span>
        <span class="hist-model">${r.model_name}</span>
        <span class="hist-ts">${fmtDate(r.timestamp)}</span>
        <button class="hist-del" onclick="deleteRecord(${r.id}, this.closest('.hist-row'))">Delete</button>`;
      listEl.appendChild(row);
    });
    container.innerHTML = "";
    container.appendChild(listEl);

    // Pagination
    const pages = Math.ceil(data.total / HIST_LIMIT);
    if (pages > 1) {
      qs("#pagination-row").style.display = "flex";
      qs("#hist-page-info").textContent = `Page ${histPage + 1} of ${pages}`;
      qs("#hist-prev-btn").disabled = histPage === 0;
      qs("#hist-next-btn").disabled = histPage >= pages - 1;
    } else {
      qs("#pagination-row").style.display = "none";
    }
  } catch {
    toast("Network error loading history.", "error");
  }
}

async function deleteRecord(id, rowEl) {
  try {
    const res  = await fetch(`/api/history/${id}`, { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      rowEl.remove();
      const total = parseInt(qs("#history-total").textContent) - 1;
      qs("#history-total").textContent = total;
      qs("#dash-history-count").textContent = fmtNum(total);
      toast("Record deleted.", "success");
    } else {
      toast(data.error || "Could not delete record.", "error");
    }
  } catch {
    toast("Network error.", "error");
  }
}

qs("#clear-history-btn").addEventListener("click", async () => {
  if (!confirm("Delete ALL prediction history? This cannot be undone.")) return;
  try {
    const res  = await fetch("/api/history?confirm=yes", { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      histPage = 0;
      toast("All history cleared.", "success");
      loadHistory();
      loadDashboard();
    } else {
      toast(data.error || "Could not clear history.", "error");
    }
  } catch {
    toast("Network error.", "error");
  }
});

qs("#hist-prev-btn").addEventListener("click", () => { if (histPage > 0) { histPage--; loadHistory(); } });
qs("#hist-next-btn").addEventListener("click", () => { histPage++; loadHistory(); });

// ── Model Performance ─────────────────────────────────────────────────────────

async function loadPerformance() {
  try {
    const res  = await fetch("/api/model-info");
    const info = await res.json();

    qs("#perf-model-name").textContent = info.model_name || "—";
    qs("#dash-model-name").textContent = info.model_name || "—";
    qs("#td-model-name").textContent   = info.model_name || "—";
    qs("#td-model-type").textContent   = info.model_type || "—";
    qs("#td-total-params").textContent = info.total_params ? fmtNum(info.total_params) : "Loading…";
    qs("#perf-total-params").textContent = info.total_params ? fmtNum(info.total_params) : "—";

    if (info.eval_results) {
      const ev = info.eval_results;
      qs("#custom-eval-card").style.display = "block";
      qs("#custom-model-name").textContent  = ev.model_name || "—";
      qs("#custom-dataset").textContent     = ev.dataset || "—";
      qs("#custom-classes").textContent     = (ev.class_names || []).join(", ") || "—";
      qs("#custom-loss").textContent        = ev.test_loss != null ? ev.test_loss : "—";
      qs("#custom-acc").textContent         = ev.test_accuracy != null ? (ev.test_accuracy * 100).toFixed(2) + "%" : "—";
      qs("#custom-note").textContent        = ev.note || "";

      // Update notice
      qs("#perf-notice").innerHTML = `<strong>✅ Custom evaluation loaded:</strong> Results from <code>training/train_cnn.py</code> are shown below. Benchmark values remain from the MobileNetV2 paper.`;
    }
  } catch { /* ignore */ }
}

// ── About: Viva Q&A ───────────────────────────────────────────────────────────

const VIVA = [
  { q: "What is a Convolutional Neural Network?",
    a: "A CNN is a deep learning architecture that uses convolutional layers with learnable filters to automatically extract spatial features from images. It is designed for grid-structured data and achieves state-of-the-art results in image classification, object detection, and segmentation." },
  { q: "Why is MobileNetV2 used in this project?",
    a: "MobileNetV2 is computationally efficient (≈3.5 M parameters, designed for mobile devices) while achieving ~72% top-1 accuracy on ImageNet. It loads quickly on CPU, making it practical for a student project demonstration without requiring a GPU." },
  { q: "What is a kernel/filter in a CNN?",
    a: "A kernel is a small weight matrix (e.g., 3×3) that slides across the input. The dot product at each position produces a feature map. The values in the kernel are learned via backpropagation." },
  { q: "What is the vanishing gradient problem?",
    a: "In deep networks, gradients can shrink exponentially as they propagate backward through many layers, making early layers learn very slowly or not at all. ReLU helps because its gradient is exactly 1 for positive inputs." },
  { q: "What is the difference between training, validation, and test data?",
    a: "Training data is used to update model weights. Validation data monitors performance during training and guides hyperparameter choices. Test data is held out until training is complete and provides an unbiased estimate of real-world accuracy." },
  { q: "How is a pretrained model different from training from scratch?",
    a: "A pretrained model has already learned general features (edges, textures, shapes) from a large dataset (ImageNet). We reuse these weights instead of initialising randomly, which requires far less data and compute for fine-tuning or direct inference." },
  { q: "What is overfitting and how is it prevented?",
    a: "Overfitting occurs when the model memorises training data and fails on unseen data (high train accuracy, low val accuracy). It is mitigated using dropout, data augmentation, early stopping, weight regularisation (L2), and using more diverse training data." },
  { q: "What is the role of Flask in this project?",
    a: "Flask serves as the HTTP backend. It loads the TensorFlow model once at startup, exposes REST API endpoints (/api/predict, /api/history, etc.), handles file uploads, runs inference, and stores results in SQLite." },
  { q: "What is Softmax and why is it used?",
    a: "Softmax converts raw output logits into a probability distribution where all values are in (0,1) and sum to 1.0. This allows interpreting the network's output as confidence scores for each class." },
  { q: "What are depthwise separable convolutions?",
    a: "Standard convolution applies a filter across all input channels simultaneously. Depthwise separable convolution separates this into: (1) a depthwise convolution applying one filter per channel, followed by (2) a 1×1 pointwise convolution to combine channels. This reduces computation by roughly 8–9× at similar accuracy." },
];

function buildViva() {
  const list = qs("#viva-list");
  VIVA.forEach((item, i) => {
    const div  = el("div", "viva-item");
    const qDiv = el("div", "viva-q",
      `<span>${i + 1}. ${item.q}</span><span class="viva-toggle">▼</span>`);
    const aDiv = el("div", "viva-a", item.a);
    qDiv.addEventListener("click", () => div.classList.toggle("open"));
    div.appendChild(qDiv);
    div.appendChild(aDiv);
    list.appendChild(div);
  });
}
buildViva();

// ── Init on load ──────────────────────────────────────────────────────────────

navigateTo("dashboard");
