import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const STYLE_ID = "h3ss-pov-stylesheet";
const NODE_NAMES = new Set([
  "MiniMaxH3FiniteSegmentSampler",
  "MiniMaxH3WindowedRefine",
]);
const EVENTS = ["h3ss_live_preview", "kj_preview_override"];

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const link = document.createElement("link");
  link.id = STYLE_ID;
  link.rel = "stylesheet";
  link.href = new URL("./live_preview.css", import.meta.url).href;
  document.head.appendChild(link);
}

function findNodeByQualifiedId(rootGraph, qid) {
  if (!rootGraph || qid == null) return null;
  const parts = String(qid).split(":");
  let graph = rootGraph;
  for (let i = 0; i < parts.length - 1; i++) {
    const parentId = parseInt(parts[i], 10);
    if (!Number.isFinite(parentId)) return null;
    const parentNode = graph?.getNodeById?.(parentId);
    if (!parentNode?.subgraph) return null;
    graph = parentNode.subgraph;
  }
  const leafId = parseInt(parts[parts.length - 1], 10);
  if (!Number.isFinite(leafId)) return null;
  return graph?.getNodeById?.(leafId) || app.graph?.getNodeById?.(qid) || null;
}

function b64ToUrl(b64, mime) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([arr], { type: mime || "image/jpeg" }));
}

function el(tag, className, parent) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (parent) parent.appendChild(node);
  return node;
}

function attachPreviewWidget(node) {
  if (node.__h3ssPreview) return;
  ensureStyles();

  const root = el("div", "h3ss-pov-root");
  const imageArea = el("div", "h3ss-pov-image-area", root);
  const img = el("img", "h3ss-pov-img", imageArea);
  const video = el("video", "h3ss-pov-video", imageArea);
  img.draggable = false;
  img.style.display = "none";
  video.muted = true;
  video.playsInline = true;
  video.loop = true;
  video.autoplay = true;
  video.style.display = "none";
  const placeholder = el("div", "h3ss-pov-placeholder", imageArea);
  placeholder.textContent = "waiting for sample…";
  const bar = el("div", "h3ss-pov-bar", root);
  const left = el("span", null, bar);
  const right = el("span", null, bar);
  left.textContent = "idle";
  right.textContent = "";

  let lastUrl = null;
  const revoke = () => {
    if (lastUrl) {
      URL.revokeObjectURL(lastUrl);
      lastUrl = null;
    }
  };

  node.__h3ssPreview = (data) => {
    try {
      if (typeof data.image === "string") {
        revoke();
        const mime = typeof data.mime === "string" ? data.mime : "image/jpeg";
        lastUrl = b64ToUrl(data.image, mime);
        placeholder.style.display = "none";
        if (mime.startsWith("video/")) {
          img.style.display = "none";
          video.style.display = "block";
          video.src = lastUrl;
          video.play?.().catch(() => {});
        } else {
          video.style.display = "none";
          img.style.display = "block";
          img.src = lastUrl;
        }
      }
      const step = data.step;
      const total = data.total;
      const sigma = data.sigma;
      const ms = data.step_ms;
      const parts = [];
      if (Number.isFinite(step) && Number.isFinite(total)) parts.push(`${step}/${total}`);
      else if (Number.isFinite(step)) parts.push(`step ${step}`);
      if (Number.isFinite(sigma)) parts.push(`σ ${sigma.toFixed(3)}`);
      left.textContent = parts.join(" · ") || "sampling";
      const size = data.w && data.h ? `${data.w}×${data.h}` : "";
      const eta = Number.isFinite(ms) && Number.isFinite(total) && Number.isFinite(step)
        ? `${((total - step) * ms / 1000).toFixed(0)}s left`
        : "";
      right.textContent = [size, eta].filter(Boolean).join(" · ");
    } catch (err) {
      console.warn("[H3 Segment Studio] live preview decode failed:", err);
    }
  };

  node.addDOMWidget("h3ss_preview", "div", root, { serialize: false });
  node.setSize([
    Math.max(node.size?.[0] ?? 360, 360),
    Math.max(node.size?.[1] ?? 480, 700),
  ]);
  const originalRemoved = node.onRemoved;
  node.onRemoved = function () {
    revoke();
    this.__h3ssPreview = null;
    return originalRemoved?.apply(this, arguments);
  };
}

for (const eventName of EVENTS) {
  api.addEventListener(eventName, (e) => {
    const data = e.detail;
    if (!data || data.node_id == null) return;
    const node = findNodeByQualifiedId(app.graph, data.node_id);
    if (typeof node?.__h3ssPreview === "function") node.__h3ssPreview(data);
  });
}

app.registerExtension({
  name: "MiniMaxH3.SegmentStudio.LivePreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData?.name)) return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      attachPreviewWidget(this);
      return result;
    };
  },
});
