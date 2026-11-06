import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorCode, errorMessage } from "../services/api";

/** Fetch on mount / when deps change. Returns { data, error, code, loading, reload }. */
export function useApi(fetcher, deps = [], { enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, code: null, loading: enabled });
  const seq = useRef(0);
  const load = useCallback(async () => {
    if (!enabled) return;
    const n = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null, code: null }));
    try {
      const data = await fetcher();
      if (n === seq.current) setState({ data, error: null, code: null, loading: false });
    } catch (err) {
      if (n === seq.current) setState((s) => ({ ...s, error: errorMessage(err), code: errorCode(err), loading: false }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled]);
  useEffect(() => { load(); }, [load]);
  return { ...state, reload: load };
}

/** Run an async action with loading/error state. Always calls the latest `fn`, so forms submit current values. */
export function useAction(fn) {
  const [state, setState] = useState({ loading: false, error: null, result: null });
  const latest = useRef(fn);
  latest.current = fn;
  const run = useCallback(async (...args) => {
    setState({ loading: true, error: null, result: null });
    try {
      const result = await latest.current(...args);
      setState({ loading: false, error: null, result });
      return result;
    } catch (err) {
      setState({ loading: false, error: errorMessage(err), result: null });
      return undefined;
    }
  }, []);
  return { ...state, run, reset: () => setState({ loading: false, error: null, result: null }) };
}

/** Server-Sent Events subscription; calls onEvent(parsed) for every message. */
export function useEventSource(url, onEvent, deps = []) {
  const handler = useRef(onEvent);
  handler.current = onEvent;
  useEffect(() => {
    if (!url) return undefined;
    const es = new EventSource(url);
    const listener = (e) => {
      try { handler.current(JSON.parse(e.data)); } catch { /* ignore keep-alives */ }
    };
    ["message", "job_progress", "job_status", "agent_start", "agent_end", "tool_call", "tool_result", "tool_error",
      "route", "run_status", "approval_required", "approval_received"].forEach((t) => es.addEventListener(t, listener));
    return () => es.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, ...deps]);
}

/** Track one background job: live progress via SSE, final record (with result) via REST. */
export function useJob(onDone) {
  const [job, setJob] = useState(null);
  const done = useRef(onDone);
  done.current = onDone;
  useEventSource(job && !["completed", "failed"].includes(job.status) ? api.jobEventsUrl() : null, async (ev) => {
    if (!job || ev.job_id !== job.id) return;
    if (ev.type === "job_progress") setJob((j) => ({ ...j, progress: ev.progress, message: ev.message, status: "running" }));
    if (ev.type === "job_status" && ["completed", "failed"].includes(ev.status)) {
      const full = await api.job(job.id);
      setJob(full);
      done.current?.(full);
    }
  }, [job?.id, job?.status]);
  // Fallback poll in case an SSE message was missed (e.g. very fast jobs).
  useEffect(() => {
    if (!job || ["completed", "failed"].includes(job.status)) return undefined;
    const t = setInterval(async () => {
      const full = await api.job(job.id).catch(() => null);
      if (full && ["completed", "failed"].includes(full.status)) { setJob(full); done.current?.(full); }
    }, 3000);
    return () => clearInterval(t);
  }, [job?.id, job?.status]);
  return [job, setJob];
}
