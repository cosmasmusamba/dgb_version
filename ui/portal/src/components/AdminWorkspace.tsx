import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, adminApi, consumeAdminEvents, isPreviewSession } from "../api";
import type { ModelRecord, StreamEvent, SystemInfo, TrainingStatus, User } from "../types";
import Icon from "./Icon";

type Theme = "light" | "dark";
type AdminPage = "overview" | "models" | "training" | "datasets" | "inference" | "users" | "keys" | "security";

interface AdminWorkspaceProps {
  user: User;
  theme: Theme;
  onToggleTheme: () => void;
  onBackToChat: () => void;
  onLogout: () => void;
}

const pageMeta: Record<AdminPage, { title: string; description: string }> = {
  overview: { title: "System overview", description: "A clear view of the services and workloads running in your DGB environment." },
  models: { title: "Model registry", description: "Review available checkpoints and manage model deployment actions." },
  training: { title: "Training jobs", description: "Monitor and control the active model training process." },
  datasets: { title: "Data pipeline", description: "Inspect pipeline progress and start a processing run." },
  inference: { title: "Inference monitoring", description: "Review recent training metrics and the live event stream." },
  users: { title: "Users & access", description: "Review your current role and the access controls enforced by the API." },
  keys: { title: "API keys & usage", description: "Review the authentication and usage capabilities currently exposed by this backend." },
  security: { title: "Security & audit", description: "Inspect live service health and understand the audit data available to this workspace." }
};

const navigation: Array<{ label: string; pages: Array<{ id: AdminPage; title: string; icon: "grid" | "book" | "settings" | "users" | "shield" | "history" | "file" }> }> = [
  { label: "WORKSPACE", pages: [{ id: "overview", title: "Overview", icon: "grid" }] },
  { label: "OPERATIONS", pages: [
    { id: "models", title: "Models", icon: "book" },
    { id: "training", title: "Training jobs", icon: "settings" },
    { id: "datasets", title: "Data pipeline", icon: "file" },
    { id: "inference", title: "Inference", icon: "history" }
  ] },
  { label: "GOVERNANCE", pages: [
    { id: "users", title: "Users & access", icon: "users" },
    { id: "keys", title: "API keys & usage", icon: "shield" },
    { id: "security", title: "Security & audit", icon: "history" }
  ] }
];

function isAdminPage(value: string): value is AdminPage {
  return value in pageMeta;
}

function display(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

function pretty(value: unknown): string {
  if (value === null || value === undefined) return "Not reported";
  return typeof value === "number" ? value.toFixed(2) : String(value);
}

export default function AdminWorkspace({
  user,
  theme,
  onToggleTheme,
  onBackToChat,
  onLogout
}: AdminWorkspaceProps) {
  const [page, setPage] = useState<AdminPage>("overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [health, setHealth] = useState<{ status: string; version: string; model: string; time: number } | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [pipeline, setPipeline] = useState<Record<string, unknown> | null>(null);
  const [metrics, setMetrics] = useState<{ epoch_history: Array<Record<string, unknown>>; step_history: Array<Record<string, unknown>> } | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [streamStatus, setStreamStatus] = useState<"connecting" | "live" | "offline">("offline");
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [warmup, setWarmup] = useState(1000);
  const [actionBusy, setActionBusy] = useState(false);

  const loadPage = useCallback(async (target: AdminPage, quiet = false) => {
    if (!quiet) {
      setLoading(true);
      setError("");
    }
    try {
      if (target === "overview") {
        const [h, s, t, m] = await Promise.all([adminApi.health(), adminApi.system(), adminApi.training(), adminApi.models()]);
        setHealth(h); setSystem(s); setTraining(t); setModels(m.models);
      } else if (target === "models") {
        const result = await adminApi.models();
        setModels(result.models);
      } else if (target === "training") {
        setTraining(await adminApi.training());
      } else if (target === "datasets") {
        setPipeline(await adminApi.pipeline());
      } else if (target === "inference") {
        setMetrics(await adminApi.metrics());
      } else if (target === "security") {
        setHealth(await adminApi.health());
        setSystem(await adminApi.system());
      }
    } catch (reason) {
      if (!quiet) setError(reason instanceof ApiError ? reason.message : "The administration service could not be reached.");
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPage(page);
    const timer = window.setInterval(() => { void loadPage(page, true); }, 20_000);
    return () => window.clearInterval(timer);
  }, [page, loadPage]);

  useEffect(() => {
    if (page !== "inference") return;
    const controller = new AbortController();
    setStreamStatus("connecting");
    void consumeAdminEvents("metrics", controller.signal, (event) => {
      setStreamStatus("live");
      setEvents((current) => [event, ...current].slice(0, 40));
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) {
        setStreamStatus("offline");
        setError(reason instanceof ApiError ? reason.message : "Could not connect to the live metrics stream.");
      }
    });
    return () => controller.abort();
  }, [page]);

  const epochRows = useMemo(() => training?.epoch_history.slice(-8).reverse() ?? [], [training]);
  const steps = metrics?.step_history ?? [];
  const newestEvent = events[0];

  async function runAction(action: () => Promise<{ status: string; pid?: number; model?: string }>, successLabel: string) {
    setActionBusy(true);
    setError("");
    setToast("");
    try {
      const result = await action();
      setToast(`${successLabel}${result.pid ? ` · process ${result.pid}` : ""}${result.model ? ` · ${result.model}` : ""}`);
      await loadPage(page, true);
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "The requested operation could not be completed.");
    } finally {
      setActionBusy(false);
    }
  }

  function choosePage(target: string) {
    if (!isAdminPage(target)) return;
    setPage(target);
    setSidebarOpen(false);
    setError("");
    setToast("");
  }

  function submitTraining(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runAction(() => adminApi.launchTraining({
      epochs,
      batch_size: batchSize,
      lr: learningRate,
      warmup
    }), "Training launch requested");
  }

  return (
    <div className={`workspace admin-workspace ${sidebarOpen ? "mobile-sidebar-open" : ""} ${isPreviewSession() ? "preview-active" : ""}`}>
      {isPreviewSession() && <div className="preview-ribbon">DEVELOPMENT PREVIEW · simulated admin data · actions do not affect the service</div>}
      {sidebarOpen && <button className="mobile-scrim" aria-label="Close administration navigation" onClick={() => setSidebarOpen(false)} />}
      <aside className="sidebar admin-sidebar">
        <div className="admin-brand-block">
          <div className="sidebar-topline"><div className="brand-lockup"><span className="brand-mark">d</span><span>DGB</span></div><span className="workspace-tag">CONTROL</span></div>
          <div className="admin-label"><Icon name="admin" size={16} /><span>Administration</span><span className="admin-status-dot" /></div>
        </div>
        <nav className="sidebar-nav admin-nav" aria-label="Administration navigation">
          {navigation.map((group) => (
            <div className="admin-nav-group" key={group.label}>
              <div className="section-label">{group.label}</div>
              {group.pages.map((item) => (
                <button key={item.id} className={page === item.id ? "nav-item active" : "nav-item"} onClick={() => choosePage(item.id)}>
                  <Icon name={item.icon} size={17} />{item.title}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button className="nav-item" onClick={onBackToChat}><Icon name="message" size={17} /> Back to conversations</button>
          <button className="nav-item" onClick={onToggleTheme}><Icon name={theme === "light" ? "moon" : "sun"} size={17} /> Switch to {theme === "light" ? "dark" : "light"} theme</button>
          <div className="account-row">
            <span className="avatar">{user.sub.slice(0, 1).toUpperCase()}</span>
            <span className="account-copy"><strong>{user.sub}</strong><small>Administrator</small></span>
            <button className="icon-button" aria-label="Sign out" onClick={onLogout}><Icon name="arrow" size={16} /></button>
          </div>
        </div>
      </aside>

      <main className="admin-main">
        <header className="admin-topbar">
          <button className="icon-button mobile-only" aria-label="Open admin navigation" onClick={() => setSidebarOpen(true)}><Icon name="menu" /></button>
          <div className="admin-breadcrumb"><span>Administration</span><Icon name="chevron" size={13} /><strong>{pageMeta[page].title}</strong></div>
          <div className="topbar-actions"><span className={`service-pill ${health?.status === "ok" ? "online" : ""}`}><i />{health?.status === "ok" ? "Service online" : "Checking service"}</span><button className="icon-button" aria-label="Refresh page data" onClick={() => void loadPage(page)}><Icon name="refresh" /></button></div>
        </header>
        <div className="admin-content">
          <div className="admin-page-heading"><div><span className="eyebrow">DGB ADMINISTRATION</span><h1>{pageMeta[page].title}</h1><p>{pageMeta[page].description}</p></div></div>
          {error && <div className="notice notice-error" role="alert"><Icon name="warning" size={17} /><span>{error}</span><button className="notice-close" aria-label="Dismiss error" onClick={() => setError("")}><Icon name="close" size={16} /></button></div>}
          {toast && <div className="notice notice-success"><Icon name="check" size={17} /><span>{toast}</span><button className="notice-close" aria-label="Dismiss message" onClick={() => setToast("")}><Icon name="close" size={16} /></button></div>}
          {loading && <div className="loading-line"><span /> Loading current service data…</div>}

          {page === "overview" && (
            <div className="admin-page-body">
              <div className="overview-banner">
                <div><span className="eyebrow">ACTIVE ENVIRONMENT</span><h2>{health?.model ?? "DGB model"} <span className="environment-badge">LOCAL</span></h2><p>Platform v{health?.version ?? "—"} · {system?.device ?? "Device status unavailable"}</p></div>
                <button className="button button-outline" onClick={() => choosePage("training")}><Icon name="settings" size={16} /> View training</button>
              </div>
              <div className="metric-card-grid">
                <MetricCard label="Training state" value={training?.is_running ? "Running" : "Idle"} detail={training?.epoch ? `Epoch ${training.epoch} of ${training.total_epochs}` : "No active training process"} icon="settings" tone={training?.is_running ? "accent" : ""} />
                <MetricCard label="Current loss" value={training?.current_loss == null ? "—" : training.current_loss.toFixed(4)} detail={training?.val_loss == null ? "Validation loss not reported" : `Validation ${training.val_loss.toFixed(4)}`} icon="grid" />
                <MetricCard label="Available models" value={models.length.toString()} detail={models.filter((model) => model.is_best).length ? "Best checkpoint is marked" : "Checkpoint inventory"} icon="book" />
                <MetricCard label="System memory" value={system ? `${system.ram_avail.toFixed(1)} GB` : "—"} detail={system ? `${system.ram_total.toFixed(1)} GB total` : "System profile unavailable"} icon="shield" />
              </div>
              <div className="admin-two-column">
                <section className="surface-card">
                  <div className="surface-heading"><div><span className="eyebrow">OPERATIONS</span><h2>Recent training progress</h2></div><button className="text-button" onClick={() => choosePage("training")}>Open jobs <Icon name="arrow" size={14} /></button></div>
                  {epochRows.length ? <div className="compact-table-wrap"><table className="data-table"><thead><tr><th>Epoch</th><th>Training loss</th><th>Validation</th></tr></thead><tbody>{epochRows.map((row, index) => <tr key={index}><td>{display(row.epoch, `Run ${epochRows.length - index}`)}</td><td>{pretty(row.avg_loss ?? row.loss)}</td><td>{pretty(row.val_loss)}</td></tr>)}</tbody></table></div> : <EmptyState title="No epoch metrics yet" detail="Metrics will appear after a training run records its first epoch." />}
                </section>
                <section className="surface-card">
                  <div className="surface-heading"><div><span className="eyebrow">INFRASTRUCTURE</span><h2>System profile</h2></div></div>
                  {system ? <div className="definition-list"><InfoRow label="Operating system" value={system.os} /><InfoRow label="Processor" value={system.cpu} /><InfoRow label="Logical cores" value={system.cores} /><InfoRow label="Accelerator" value={system.gpu || "Not detected"} /><InfoRow label="Recommended device" value={system.device} /></div> : <EmptyState title="System profile unavailable" detail="The protected system endpoint did not return a profile." />}
                </section>
              </div>
              <section className="surface-card capability-card">
                <div className="surface-heading"><div><span className="eyebrow">PLATFORM CAPABILITIES</span><h2>What this service exposes</h2></div></div>
                <div className="capability-grid"><Capability icon="book" title="Model registry" detail="Browse checkpoints and request deploy or rollback actions." onClick={() => choosePage("models")} /><Capability icon="history" title="Live inference metrics" detail="Review recent metrics and authenticated event-stream updates." onClick={() => choosePage("inference")} /><Capability icon="users" title="Access controls" detail="Review the authenticated role and protected service boundaries." onClick={() => choosePage("users")} /></div>
              </section>
            </div>
          )}

          {page === "models" && (
            <div className="admin-page-body">
              <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">CHECKPOINT INVENTORY</span><h2>Available model files</h2></div><span className="count-badge">{models.length} models</span></div>
                {models.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Model checkpoint</th><th>Size</th><th>Epoch</th><th>Loss</th><th>Status</th><th>Actions</th></tr></thead><tbody>{models.map((item) => <tr key={item.name}><td><div className="model-name"><span className="model-file-icon"><Icon name="file" size={16} /></span><span><strong>{item.name}</strong><small>Version {item.ver || "unversioned"}</small></span></div></td><td>{item.size_mb.toFixed(1)} MB</td><td>{display(item.epoch)}</td><td>{item.loss == null ? "—" : item.loss.toFixed(4)}</td><td><span className={`table-status ${item.is_best ? "best" : ""}`}><i />{item.is_best ? "Best checkpoint" : "Available"}</span></td><td><div className="table-actions"><button className="button button-small button-outline" disabled={actionBusy} onClick={() => { if (window.confirm(`Request deployment of ${item.name}? The backend currently queues this action.`)) void runAction(() => adminApi.deployModel(item.name), "Deployment queued"); }}>Deploy</button><button className="button button-small button-quiet" disabled={actionBusy} onClick={() => { if (window.confirm(`Request rollback to ${item.name}? The backend currently queues this action.`)) void runAction(() => adminApi.rollbackModel(item.name), "Rollback queued"); }}>Rollback</button></div></td></tr>)}</tbody></table></div> : <EmptyState title="No checkpoints found" detail="The model registry currently has no .pt files in the configured model directory." />}
              </section>
              <p className="footnote"><Icon name="warning" size={14} /> Deploy and rollback endpoints currently acknowledge requests as queued; they do not activate a checkpoint in the inference process.</p>
            </div>
          )}

          {page === "training" && (
            <div className="admin-page-body">
              <div className="training-state-card"><div className={`status-emblem ${training?.is_running ? "running" : ""}`}><Icon name="settings" size={21} /></div><div className="training-state-copy"><span className="eyebrow">TRAINING PROCESS</span><h2>{training?.is_running ? "A training job is in progress" : "No training job is active"}</h2><p>{training?.is_running ? `Epoch ${training.epoch ?? "—"} · Running for ${training.elapsed_sec?.toFixed(0) ?? "—"} seconds` : "Start a new run with the current configured model and dataset."}</p></div><div className="training-state-actions"><button className="button button-outline" onClick={() => void loadPage("training")}><Icon name="refresh" size={15} /> Refresh</button>{training?.is_running && <button className="button button-danger" disabled={actionBusy} onClick={() => { if (window.confirm("Stop the active training process?")) void runAction(() => adminApi.stopTraining(), "Stop requested"); }}>Stop training</button>}</div></div>
              <div className="admin-two-column training-columns"><section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">CURRENT RUN</span><h2>Run metrics</h2></div></div><div className="training-metrics"><MetricLine label="Training loss" value={training?.current_loss == null ? "—" : training.current_loss.toFixed(4)} /><MetricLine label="Validation loss" value={training?.val_loss == null ? "—" : training.val_loss.toFixed(4)} /><MetricLine label="Perplexity" value={training?.perplexity == null ? "—" : training.perplexity.toFixed(2)} /><MetricLine label="Progress" value={training?.epoch ? `${training.epoch} / ${training.total_epochs} epochs` : "Not started"} /></div></section>
                <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">NEW RUN</span><h2>Launch training</h2></div></div><form className="launch-form" onSubmit={submitTraining}><label>Epochs<input type="number" min="1" max="300" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} required /></label><label>Batch size<input type="number" min="1" max="512" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} required /></label><label>Learning rate<input type="number" min="0.000001" max="0.1" step="any" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} required /></label><label>Warmup steps<input type="number" min="0" value={warmup} onChange={(event) => setWarmup(Number(event.target.value))} required /></label><p className="form-help">The backend launches <code>model_trainer.py</code>. These values are accepted for now; see the service logs for the effective loaded configuration.</p><button className="button button-primary" type="submit" disabled={actionBusy || !!training?.is_running}><Icon name="settings" size={16} /> {actionBusy ? "Submitting…" : "Launch training job"}</button></form></section></div>
              <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">HISTORY</span><h2>Recent epochs</h2></div></div>{epochRows.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Epoch</th><th>Average loss</th><th>Validation loss</th><th>Perplexity</th></tr></thead><tbody>{epochRows.map((row, index) => <tr key={index}><td>{display(row.epoch)}</td><td>{pretty(row.avg_loss ?? row.loss)}</td><td>{pretty(row.val_loss)}</td><td>{pretty(row.perplexity)}</td></tr>)}</tbody></table></div> : <EmptyState title="No completed epochs" detail="Training history appears here when metrics have been written." />}</section>
            </div>
          )}

          {page === "datasets" && (
            <div className="admin-page-body">
              <section className="surface-card pipeline-card"><div className="surface-heading"><div><span className="eyebrow">DATA OPERATIONS</span><h2>Pipeline state</h2></div><button className="button button-outline" onClick={() => void loadPage("datasets")}><Icon name="refresh" size={15} /> Refresh</button></div>
                {pipeline ? <div className="pipeline-data"><pre>{JSON.stringify(pipeline, null, 2)}</pre></div> : <EmptyState title="Pipeline state unavailable" detail="The API did not return a pipeline state. Check that the configured pipeline has written a state snapshot." />}
                <div className="pipeline-action"><div><strong>Start a data pipeline run</strong><p>Runs the backend's configured <code>main_pipeline.py</code> process.</p></div><button className="button button-primary" disabled={actionBusy} onClick={() => { if (window.confirm("Start the configured data pipeline now?")) void runAction(() => adminApi.runPipeline(), "Pipeline start requested"); }}><Icon name="arrow" size={15} /> Run pipeline</button></div>
              </section>
              <div className="notice notice-info"><Icon name="shield" size={17} /><span>Dataset inventory, source-level counts, and per-dataset permissions are not exposed by the current API. This view only displays the real pipeline status returned by the service.</span></div>
            </div>
          )}

          {page === "inference" && (
            <div className="admin-page-body">
              <div className="stream-banner"><div className="stream-live-mark"><i /></div><div><strong>Metrics event stream</strong><p>Authenticated updates from <code>/stream/metrics</code></p></div><span className={`table-status ${streamStatus === "live" ? "best" : ""}`}><i />{streamStatus}</span></div>
              <div className="metric-card-grid three-cards"><MetricCard label="Recent step records" value={steps.length.toString()} detail="Loaded from metrics history" icon="history" /><MetricCard label="Latest event" value={newestEvent ? new Date(newestEvent.timestamp * 1000).toLocaleTimeString() : "—"} detail={newestEvent?.type ?? "No live event received"} icon="refresh" /><MetricCard label="Training loss" value={training?.current_loss == null ? "—" : training.current_loss.toFixed(4)} detail="Most recent epoch record" icon="grid" /></div>
              <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">RECENT METRICS</span><h2>Training steps</h2></div></div>{steps.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Step</th><th>Loss</th><th>Learning rate</th><th>Gradient norm</th><th>Elapsed</th></tr></thead><tbody>{steps.slice(-40).reverse().map((row, index) => <tr key={index}><td>{display(row.step ?? row.global_step, `—`)}</td><td>{pretty(row.loss)}</td><td>{pretty(row.lr ?? row.learning_rate)}</td><td>{pretty(row.grad_norm)}</td><td>{pretty(row.elapsed_sec)}</td></tr>)}</tbody></table></div> : <EmptyState title="No step metrics available" detail="Step history will appear when the training logger writes metric records." />}</section>
              <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">LIVE EVENT FEED</span><h2>Latest updates</h2></div><span className="count-badge">{events.length} received</span></div>{events.length ? <div className="event-list">{events.slice(0, 12).map((event, index) => <div className="event-row" key={`${event.timestamp}-${index}`}><span className={`event-type event-${event.type}`}>{event.type}</span><span>{display(event.payload.message ?? event.payload.loss ?? JSON.stringify(event.payload))}</span><time>{new Date(event.timestamp * 1000).toLocaleTimeString()}</time></div>)}</div> : <EmptyState title="Waiting for metrics" detail="Events arrive while a backend process is publishing training metrics." />}</section>
              <div className="notice notice-info"><Icon name="warning" size={17} /><span>Inference request counts and token usage are not currently emitted by the backend, so this page reports training metrics only.</span></div>
            </div>
          )}

          {page === "users" && <GovernancePage kind="users" user={user} />}
          {page === "keys" && <GovernancePage kind="keys" user={user} />}
          {page === "security" && <GovernancePage kind="security" user={user} health={health} system={system} />}
        </div>
      </main>
    </div>
  );
}

function MetricCard({ label, value, detail, icon, tone = "" }: { label: string; value: string; detail: string; icon: "settings" | "grid" | "book" | "shield" | "history" | "refresh" | "users"; tone?: string }) {
  return <article className={`metric-card ${tone}`}><div className="metric-card-top"><span>{label}</span><span className="metric-icon"><Icon name={icon} size={17} /></span></div><strong>{value}</strong><small>{detail}</small></article>;
}

function InfoRow({ label, value }: { label: string; value: unknown }) {
  return <div className="definition-row"><span>{label}</span><strong>{display(value)}</strong></div>;
}

function MetricLine({ label, value }: { label: string; value: string }) {
  return <div className="metric-line"><span>{label}</span><strong>{value}</strong></div>;
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-state"><span className="empty-state-icon"><Icon name="file" size={19} /></span><strong>{title}</strong><p>{detail}</p></div>;
}

function Capability({ icon, title, detail, onClick }: { icon: "book" | "history" | "users"; title: string; detail: string; onClick: () => void }) {
  return <button className="capability-item" onClick={onClick}><span className="capability-icon"><Icon name={icon} size={18} /></span><span><strong>{title}</strong><small>{detail}</small></span><Icon name="chevron" size={15} /></button>;
}

function GovernancePage({
  kind,
  user,
  health,
  system
}: {
  kind: "users" | "keys" | "security";
  user: User;
  health?: { status: string; version: string; model: string; time: number } | null;
  system?: SystemInfo | null;
}) {
  if (kind === "users") {
    return <div className="admin-page-body">
      <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">IDENTITY</span><h2>Signed-in administrator</h2></div><span className="role-pill role-admin">ADMIN</span></div><div className="identity-card"><span className="avatar avatar-large">{user.sub.slice(0, 1).toUpperCase()}</span><div><strong>{user.sub}</strong><p>Role is read from <code>/auth/me</code> and enforced by the API on every administrator endpoint.</p></div></div></section>
      <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">ACCESS MODEL</span><h2>Current role boundaries</h2></div></div><div className="role-table"><div className="role-row role-header"><span>Role</span><span>Inference</span><span>Administration</span></div><div className="role-row"><strong><span className="role-pill">readonly</span></strong><span>Read-only endpoints</span><span>Blocked</span></div><div className="role-row"><strong><span className="role-pill">user</span></strong><span>Authenticated inference</span><span>Blocked</span></div><div className="role-row"><strong><span className="role-pill role-admin">admin</span></strong><span>Authenticated inference</span><span>Protected admin routes</span></div></div></section>
      <div className="notice notice-info"><Icon name="warning" size={17} /><span>User listing, invitations, account changes, and role assignment are not exposed by this backend. Public registration is disabled. User provisioning remains a backend/deployment operation; no synthetic user rows or unsafe role controls are shown here.</span></div>
    </div>;
  }
  if (kind === "keys") {
    return <div className="admin-page-body">
      <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">AUTHENTICATION</span><h2>API access methods</h2></div><span className="service-pill online"><i /> JWT authentication</span></div><div className="key-info-grid"><div><span>Interactive access</span><strong>Username and password</strong><small>POST /auth/login returns a short-lived access token and refresh token.</small></div><div><span>Protected API requests</span><strong>Bearer access token</strong><small>The UI keeps tokens in this browser session and refreshes access when needed.</small></div><div><span>Public registration</span><strong>Disabled</strong><small>New accounts are not created by this interface.</small></div></div></section>
      <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">USAGE VISIBILITY</span><h2>What is measurable today</h2></div></div><div className="definition-list"><InfoRow label="Training metrics endpoint" value="Available to admins" /><InfoRow label="Inference token usage endpoint" value="Not available" /><InfoRow label="API key lifecycle endpoint" value="Not available" /><InfoRow label="Current access role" value={user.role} /></div></section>
      <div className="notice notice-info"><Icon name="shield" size={17} /><span>The current API has no API-key creation/revocation or per-user usage endpoint. JWTs are not displayed or copied into this page. Token-usage billing and key management need backend support before they can be managed safely.</span></div>
    </div>;
  }
  return <div className="admin-page-body">
    <div className="metric-card-grid three-cards"><MetricCard label="API health" value={health?.status ?? "—"} detail={`Version ${health?.version ?? "unknown"}`} icon="shield" tone={health?.status === "ok" ? "accent" : ""} /><MetricCard label="Runtime model" value={health?.model ?? "—"} detail={system?.device ? `Device: ${system.device}` : "System profile unavailable"} icon="book" /><MetricCard label="Administrator access" value={user.role === "admin" ? "Verified" : "Denied"} detail={`Authenticated as ${user.sub}`} icon="users" /></div>
    <section className="surface-card"><div className="surface-heading"><div><span className="eyebrow">SERVICE SECURITY</span><h2>Operational signals</h2></div></div><div className="definition-list"><InfoRow label="Health endpoint" value="GET /v1/admin/health · admin required" /><InfoRow label="Role verification" value="Server-side role dependency" /><InfoRow label="Registration endpoint" value="Disabled (403)" /><InfoRow label="Live metric stream" value="Authenticated bearer token" /><InfoRow label="Operating system" value={system?.os} /><InfoRow label="Recommended device" value={system?.device} /></div></section>
    <div className="notice notice-info"><Icon name="warning" size={17} /><span>Security-event audit logs, user login history, and API-key activity are not exposed by the current backend. This page reports only live health and access facts available from existing endpoints.</span></div>
  </div>;
}
