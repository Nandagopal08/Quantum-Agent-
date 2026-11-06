import axios from "axios";

// Same-origin "/api" in development (Vite proxy) and when served by FastAPI.
// Set VITE_API_URL (e.g. a Colab tunnel such as https://xxxx.trycloudflare.com/api) to target another host.
export const API_BASE = import.meta.env.VITE_API_URL || "/api";

// Optional API token (required when the backend runs behind a public Colab tunnel). Opening the app with
// ?token=... stores it for this browser.
const TOKEN_KEY = "qadt_api_token";
function readToken() {
  try {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get("token");
    if (fromUrl) {
      localStorage.setItem(TOKEN_KEY, fromUrl);
      params.delete("token");     // keep the secret out of the address bar, history and bookmarks
      const q = params.toString();
      window.history.replaceState(null, "", `${window.location.pathname}${q ? `?${q}` : ""}${window.location.hash}`);
    }
    return fromUrl || localStorage.getItem(TOKEN_KEY) || "";
  } catch { return ""; }
}
const token = readToken();
const withToken = (url) => (token ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : url);

export const http = axios.create({ baseURL: API_BASE, timeout: 300_000,
  headers: token ? { Authorization: `Bearer ${token}` } : {} });

export function errorMessage(err) {
  const d = err?.response?.data;
  if (d?.detail) return typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
  if (err?.code === "ERR_NETWORK") return `Cannot reach the backend at ${API_BASE}. Is the FastAPI server running?`;
  return err?.message || String(err);
}

export const errorCode = (err) => err?.response?.data?.error || null;

const get = (url, params) => http.get(url, { params }).then((r) => r.data);
const post = (url, body, params) => http.post(url, body, { params }).then((r) => r.data);

export const api = {
  health: () => get("/health"),
  overview: () => get("/overview"),
  storage: () => get("/storage"),
  initStorage: () => post("/storage/init"),
  settings: () => get("/settings"),
  saveSettings: (patch) => http.put("/settings", patch).then((r) => r.data),
  resetSettings: () => post("/settings/reset"),
  llmStatus: () => get("/settings/llm-status"),

  datasets: () => get("/datasets"),
  downloadDataset: (name, force = false) => post(`/datasets/${name}/download`, null, { force }),
  uploadDataset: (name, file, onProgress) => {
    const form = new FormData();
    form.append("file", file);
    return http.post(`/datasets/${name}/upload`, form, {
      onUploadProgress: (e) => onProgress?.(e.total ? e.loaded / e.total : 0),
    }).then((r) => r.data);
  },
  cmapssEda: (subset, refresh = false) => get(`/analysis/cmapss/${subset}`, { refresh }),
  cmapssPipeline: (subset) => get(`/analysis/cmapss/${subset}/pipeline`),
  imsEda: (test) => get(`/analysis/ims/${test}`),
  imsFeatures: (test) => post(`/analysis/ims/${test}/features`),

  models: () => get("/models"),
  trainRul: (body) => post("/training/rul", body),
  trainAnomaly: (body) => post("/training/anomaly", body),
  trainingLog: (run) => get(`/training/logs/${run}`),
  jobs: (params) => get("/jobs", params),
  job: (id) => get(`/jobs/${id}`),

  quantumRun: (body) => post("/quantum/run", body),
  quantumExperiments: () => get("/quantum/experiments"),

  qseqReport: (subset = "FD001") => get("/qseq/report", { subset }),
  qseqBenchmark: (body) => post("/qseq/benchmark", body),
  qseqQiskit: (subset = "FD001") => get("/qseq/qiskit", { subset }),
  qseqStartRun: (body) => post("/qseq/agents/runs", body),
  qseqAgentStatus: () => get("/qseq/agents/status"),
  qseqRuns: () => get("/qseq/agents/runs"),
  qseqRun: (id) => get(`/qseq/agents/runs/${id}`),
  qseqRunEventsUrl: (id) => withToken(`${API_BASE}/qseq/agents/runs/${id}/events`),

  fleet: (subset, model) => get("/fleet/turbofan", { subset, model }),
  bearings: (test, model, as_of) => get("/fleet/bearings", { test, model, as_of }),
  prediction: (id, params) => get(`/assets/${id}/prediction`, params),

  twinState: (id, params) => get(`/twin/${id}/state`, params),
  twinPlayback: (id, params) => get(`/twin/${id}/playback`, params),
  twinSimulate: (id, body) => post(`/twin/${id}/simulate`, body),
  recordMaintenance: (id, body) => post(`/twin/${id}/maintenance`, body),
  maintenanceEvents: (asset_id) => get("/maintenance/events", { asset_id }),

  counterfactual: (body) => post("/counterfactual", body),
  optimize: (body) => post("/optimization/schedule", body),
  optimizationResults: () => get("/optimization/results"),

  explainAsset: (id, model) => get(`/explain/asset/${id}`, { model }),
  explainModel: (body) => post("/explain/model", body),
  explainReports: () => get("/explain/reports"),
  fileUrl: (area, name) => withToken(`${API_BASE}/files/${area}/${encodeURIComponent(name)}`),

  comparisons: () => get("/experiments/comparisons"),

  agentSpecs: () => get("/agents/specs"),
  startRun: (body) => post("/agents/runs", body),
  runs: () => get("/agents/runs"),
  run: (id) => get(`/agents/runs/${id}`),
  approve: (id, body) => post(`/agents/runs/${id}/approval`, body),
  ask: (question) => post("/agents/ask", { question }),
  runEventsUrl: (id) => withToken(`${API_BASE}/agents/runs/${id}/events`),
  jobEventsUrl: () => withToken(`${API_BASE}/events/jobs`),
};
