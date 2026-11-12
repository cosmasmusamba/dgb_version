import { FormEvent, KeyboardEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, clearSession, completeChat, isPreviewSession } from "../api";
import { userStorage } from "../storage";
import type { ChatMessage, ChatThread, Project, User } from "../types";
import Icon from "./Icon";

type Theme = "light" | "dark";

interface ChatWorkspaceProps {
  user: User;
  theme: Theme;
  onToggleTheme: () => void;
  onOpenAdmin: () => void;
  onLogout: () => void;
}

const starterPrompts = [
  { title: "Think something through", text: "Help me think through an idea I have." },
  { title: "Make a plan", text: "Help me turn my goal into a clear, practical plan." },
  { title: "Work with a document", text: "I have a document I would like to explore with you." }
];

function makeId(): string {
  return crypto.randomUUID();
}

function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(timestamp);
}

function downloadArtifact(message: ChatMessage, title: string) {
  const blob = new Blob([`# ${title}\n\n${message.content}\n`], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "dgb-artifact"}.md`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function ChatWorkspace({
  user,
  theme,
  onToggleTheme,
  onOpenAdmin,
  onLogout
}: ChatWorkspaceProps) {
  const [threads, setThreads] = useState<ChatThread[]>(() => userStorage.threads(user.sub));
  const [projects, setProjects] = useState<Project[]>(() => userStorage.projects(user.sub));
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [projectFilter, setProjectFilter] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("dgb");
  const [attachments, setAttachments] = useState<Array<{ name: string; content: string }>>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [contextOpen, setContextOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [projectDialogOpen, setProjectDialogOpen] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const aborter = useRef<AbortController | null>(null);

  useEffect(() => userStorage.saveThreads(user.sub, threads), [user.sub, threads]);
  useEffect(() => userStorage.saveProjects(user.sub, projects), [user.sub, projects]);
  useEffect(() => {
    const stopResponse = () => aborter.current?.abort();
    window.addEventListener("dgb:stop", stopResponse);
    return () => window.removeEventListener("dgb:stop", stopResponse);
  }, []);

  const activeThread = threads.find((thread) => thread.id === activeThreadId) ?? null;
  const recentThreads = useMemo(() => {
    const filtered = projectFilter
      ? threads.filter((thread) => thread.projectId === projectFilter)
      : threads;
    return filtered.slice(0, 30);
  }, [threads, projectFilter]);
  const activeProject = projects.find((project) => project.id === projectFilter);

  function updateThread(threadId: string, update: (thread: ChatThread) => ChatThread) {
    setThreads((current) => current.map((thread) => thread.id === threadId ? update(thread) : thread));
  }

  function newThread(projectId = projectFilter) {
    const thread: ChatThread = {
      id: makeId(),
      title: "New conversation",
      projectId,
      updatedAt: Date.now(),
      messages: []
    };
    setThreads((current) => [thread, ...current]);
    setActiveThreadId(thread.id);
    setPrompt("");
    setAttachments([]);
    setError("");
    setSidebarOpen(false);
    window.setTimeout(() => composer.current?.focus(), 0);
  }

  function createProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = projectName.trim();
    if (!name) return;
    const project: Project = { id: makeId(), name, createdAt: Date.now() };
    setProjects((current) => [project, ...current]);
    setProjectFilter(project.id);
    setProjectName("");
    setProjectDialogOpen(false);
    newThread(project.id);
  }

  function removeThread(event: MouseEvent, threadId: string) {
    event.stopPropagation();
    setThreads((current) => current.filter((thread) => thread.id !== threadId));
    if (activeThreadId === threadId) setActiveThreadId(null);
  }

  async function addFiles(files: FileList | null) {
    if (!files) return;
    const supported = Array.from(files);
    for (const file of supported) {
      if (!/\.(txt|md|csv|json|log)$/i.test(file.name)) {
        setError(`${file.name}: attach a text, Markdown, CSV, JSON, or log file.`);
        continue;
      }
      if (file.size > 1_000_000) {
        setError(`${file.name}: files must be smaller than 1 MB.`);
        continue;
      }
      const content = await file.text();
      setAttachments((current) => [...current, { name: file.name, content }]);
      setError("");
    }
    if (fileInput.current) fileInput.current.value = "";
  }

  async function sendMessage(text = prompt) {
    const body = text.trim();
    if (!body || busy) return;
    let thread = activeThread;
    if (!thread) {
      thread = {
        id: makeId(),
        title: body.slice(0, 54),
        projectId: projectFilter,
        updatedAt: Date.now(),
        messages: []
      };
      setThreads((current) => [thread as ChatThread, ...current]);
      setActiveThreadId(thread.id);
    }
    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: body,
      createdAt: Date.now(),
      attachments: attachments.map(({ name, content }) => ({ name, content }))
    };
    const assistantMessage: ChatMessage = {
      id: makeId(),
      role: "assistant",
      content: "",
      createdAt: Date.now()
    };
    const attachmentContext = attachments.map((file) => `\n\n[Attached file: ${file.name}]\n${file.content}`).join("");
    const modelMessages = [
      ...thread.messages.map(({ role, content }) => ({ role, content })),
      { role: "user", content: `${body}${attachmentContext}` }
    ];
    updateThread(thread.id, (current) => ({
      ...current,
      title: current.messages.length ? current.title : body.slice(0, 54),
      updatedAt: Date.now(),
      messages: [...current.messages, userMessage, assistantMessage]
    }));
    setPrompt("");
    setAttachments([]);
    setBusy(true);
    setError("");
    aborter.current = new AbortController();
    try {
      await completeChat(model, modelMessages, aborter.current.signal, (fragment) => {
        updateThread(thread!.id, (current) => ({
          ...current,
          updatedAt: Date.now(),
          messages: current.messages.map((message) => message.id === assistantMessage.id
            ? { ...message, content: message.content + fragment }
            : message)
        }));
      });
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === "AbortError")) {
        const message = reason instanceof ApiError ? reason.message : "The model could not finish this response.";
        updateThread(thread.id, (current) => ({
          ...current,
          messages: current.messages.map((item) => item.id === assistantMessage.id
            ? { ...item, content: item.content ? `${item.content}\n\n_${message}_` : `_${message}_` }
            : item)
        }));
        setError(message);
      }
    } finally {
      setBusy(false);
      aborter.current = null;
    }
  }

  function handleComposerKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  const allArtifacts = threads.flatMap((thread) =>
    thread.messages.filter((message) => message.role === "assistant" && message.content.trim())
      .map((message) => ({ message, title: thread.title, threadId: thread.id }))
  );

  return (
    <div className={`workspace chat-workspace ${sidebarOpen ? "mobile-sidebar-open" : ""} ${isPreviewSession() ? "preview-active" : ""}`}>
      {isPreviewSession() && <div className="preview-ribbon">DEVELOPMENT PREVIEW · local fixtures · no backend requests</div>}
      {sidebarOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
      <aside className="sidebar chat-sidebar">
        <div className="sidebar-topline">
          <div className="brand-lockup"><span className="brand-mark">d</span><span>DGB</span></div>
          <button className="icon-button mobile-only" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}><Icon name="close" /></button>
          <span className="workspace-tag">WORKSPACE</span>
        </div>
        <button className="new-chat-button" onClick={() => newThread()}><Icon name="plus" size={17} /> New conversation</button>
        <nav className="sidebar-nav" aria-label="Workspace navigation">
          <button className={!projectFilter ? "nav-item active" : "nav-item"} onClick={() => { setProjectFilter(null); setActiveThreadId(null); }}>
            <Icon name="message" size={17} /> Conversations
          </button>
          <button className="nav-item" onClick={() => setProjectDialogOpen(true)}><Icon name="project" size={17} /> Create a project <span className="nav-trailing"><Icon name="plus" size={14} /></span></button>
        </nav>
        {projects.length > 0 && (
          <div className="sidebar-section">
            <div className="section-label">Your projects</div>
            {projects.map((project) => (
              <button key={project.id} className={projectFilter === project.id ? "nav-item active" : "nav-item"} onClick={() => { setProjectFilter(project.id); setActiveThreadId(null); setSidebarOpen(false); }}>
                <Icon name="project" size={16} /><span className="truncate">{project.name}</span>
              </button>
            ))}
          </div>
        )}
        <div className="sidebar-section conversation-list">
          <div className="section-label">{activeProject ? activeProject.name : "Recent"}</div>
          {recentThreads.length === 0
            ? <p className="sidebar-empty">{projectFilter ? "No conversations in this project yet." : "Your conversations will appear here."}</p>
            : recentThreads.map((thread) => (
              <div key={thread.id} className={`history-item ${activeThreadId === thread.id ? "selected" : ""}`}>
                <button className="history-select" onClick={() => { setActiveThreadId(thread.id); setSidebarOpen(false); }}>
                  <Icon name="message" size={14} /><span className="truncate">{thread.title}</span>
                </button>
                <button className="history-remove" aria-label={`Delete ${thread.title}`} onClick={(event) => removeThread(event, thread.id)}><Icon name="trash" size={14} /></button>
              </div>
            ))}
        </div>
        <div className="sidebar-bottom">
          <button className="nav-item" onClick={onToggleTheme}><Icon name={theme === "light" ? "moon" : "sun"} size={17} /> Switch to {theme === "light" ? "dark" : "light"} theme</button>
          {user.role === "admin" && <button className="nav-item admin-entry" onClick={onOpenAdmin}><Icon name="admin" size={17} /> Administration <Icon name="arrow" size={14} /></button>}
          <div className="account-row">
            <span className="avatar">{user.sub.slice(0, 1).toUpperCase()}</span>
            <span className="account-copy"><strong>{user.sub}</strong><small>{user.role} account</small></span>
            <div className="account-menu-wrap">
              <button className="icon-button" aria-label="Account options" onClick={() => setMenuOpen((open) => !open)}><Icon name="more" /></button>
              {menuOpen && <div className="floating-menu"><button onClick={() => { clearSession(); onLogout(); }}>Sign out</button></div>}
            </div>
          </div>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-topbar">
          <button className="icon-button mobile-only" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Icon name="menu" /></button>
          <div className="model-select-wrap">
            <select aria-label="Select model" value={model} onChange={(event) => setModel(event.target.value)}>
              <option value="dgb">DGB · active workspace model</option>
            </select>
            <Icon name="chevronDown" size={14} />
          </div>
          <div className="topbar-actions">
            {activeThread && <span className="saved-indicator"><Icon name="check" size={14} /> Saved on this device</span>}
            <button className={`icon-button ${contextOpen ? "active" : ""}`} aria-label="Toggle conversation details" onClick={() => setContextOpen((open) => !open)}><Icon name="panel" /></button>
          </div>
        </header>

        <div className={`chat-content ${contextOpen ? "with-context" : ""}`}>
          <section className="conversation-column">
            {activeThread ? (
              <>
                <div className="message-scroll">
                  <div className="conversation-title">
                    {activeProject && <span className="project-breadcrumb"><Icon name="project" size={14} /> {activeProject.name}</span>}
                    <h1>{activeThread.title}</h1>
                  </div>
                  {activeThread.messages.map((message) => (
                    <article key={message.id} className={`message-row ${message.role}`}>
                      {message.role === "assistant"
                        ? <span className="assistant-mark">d</span>
                        : <span className="user-avatar">{user.sub.slice(0, 1).toUpperCase()}</span>}
                      <div className="message-body">
                        <div className="message-author">{message.role === "assistant" ? "DGB" : user.sub}<time>{formatTime(message.createdAt)}</time></div>
                        {message.attachments?.map((file) => <div className="message-file" key={file.name}><Icon name="file" size={15} />{file.name}</div>)}
                        <div className="message-text">{message.content || (busy && message.role === "assistant" ? <span className="typing-indicator"><i /><i /><i /></span> : "")}</div>
                        {message.role === "assistant" && message.content.trim() && (
                          <div className="message-tools">
                            <button onClick={() => void navigator.clipboard.writeText(message.content)}><Icon name="copy" size={14} /> Copy</button>
                            <button onClick={() => downloadArtifact(message, activeThread.title)}><Icon name="download" size={14} /> Save as artifact</button>
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                  {error && <div className="inline-notice"><Icon name="warning" size={15} /> {error}</div>}
                </div>
                <Composer
                  prompt={prompt}
                  setPrompt={setPrompt}
                  attachments={attachments}
                  removeAttachment={(name) => setAttachments((items) => items.filter((item) => item.name !== name))}
                  onAttach={() => fileInput.current?.click()}
                  onFiles={addFiles}
                  onSubmit={() => void sendMessage()}
                  onKeyDown={handleComposerKey}
                  busy={busy}
                  composer={composer}
                  fileInput={fileInput}
                />
              </>
            ) : (
              <div className="welcome-panel">
                <div className="welcome-orbit"><span className="brand-mark">d</span></div>
                <p className="eyebrow">DGB AI · YOUR THINKING PARTNER</p>
                <h1>What would you like<br />to explore today?</h1>
                <p className="welcome-description">A thoughtful space for ideas, questions, and the work in between.</p>
                <Composer
                  prompt={prompt}
                  setPrompt={setPrompt}
                  attachments={attachments}
                  removeAttachment={(name) => setAttachments((items) => items.filter((item) => item.name !== name))}
                  onAttach={() => fileInput.current?.click()}
                  onFiles={addFiles}
                  onSubmit={() => void sendMessage()}
                  onKeyDown={handleComposerKey}
                  busy={busy}
                  composer={composer}
                  fileInput={fileInput}
                />
                <div className="starter-grid">
                  {starterPrompts.map((item) => (
                    <button className="starter-card" key={item.title} onClick={() => { setPrompt(item.text); composer.current?.focus(); }}>
                      <span>{item.title}</span><Icon name="arrow" size={15} />
                    </button>
                  ))}
                </div>
                <p className="local-data-note"><Icon name="shield" size={14} /> Conversation history and projects are saved in this browser.</p>
              </div>
            )}
          </section>

          {contextOpen && (
            <aside className="context-panel">
              <div className="context-heading"><div><span className="eyebrow">WORKSPACE</span><h2>Conversation details</h2></div><button className="icon-button" aria-label="Close details" onClick={() => setContextOpen(false)}><Icon name="close" /></button></div>
              <div className="context-section"><h3>In this conversation</h3><div className="context-item"><Icon name="message" /><span>{activeThread?.messages.length ?? 0} messages</span></div><div className="context-item"><Icon name="project" /><span>{activeProject?.name ?? "Personal workspace"}</span></div></div>
              <div className="context-section"><div className="context-heading-inline"><h3>Attachments</h3><span>{activeThread?.messages.reduce((count, item) => count + (item.attachments?.length ?? 0), 0) ?? 0}</span></div>
                {activeThread?.messages.flatMap((message) => message.attachments ?? []).length
                  ? activeThread.messages.flatMap((message) => message.attachments ?? []).map((file, index) => <div className="context-file" key={`${file.name}-${index}`}><Icon name="file" size={16} /><span className="truncate">{file.name}</span></div>)
                  : <p className="muted-copy">Files you attach to this conversation will show up here.</p>}
              </div>
              <div className="context-section"><div className="context-heading-inline"><h3>Artifacts</h3><span>{activeThread?.messages.filter((message) => message.role === "assistant" && message.content.trim()).length ?? 0}</span></div>
                {activeThread?.messages.filter((message) => message.role === "assistant" && message.content.trim()).map((message) => <button className="artifact-card" key={message.id} onClick={() => downloadArtifact(message, activeThread.title)}><Icon name="file" size={17} /><span><strong>{activeThread.title}</strong><small>Markdown · save to your device</small></span><Icon name="download" size={15} /></button>)}
                {!activeThread?.messages.some((message) => message.role === "assistant" && message.content.trim()) && <p className="muted-copy">Save an answer as an artifact to keep useful work close by.</p>}
              </div>
              {allArtifacts.length > 0 && <div className="context-footnote">Artifacts are generated from your answers and downloaded locally. DGB does not upload them.</div>}
            </aside>
          )}
        </div>
      </main>

      {projectDialogOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setProjectDialogOpen(false); }}>
          <form className="modal-card" onSubmit={createProject}>
            <div className="modal-heading"><div><span className="eyebrow">YOUR WORKSPACE</span><h2>Create a project</h2></div><button type="button" className="icon-button" aria-label="Close dialog" onClick={() => setProjectDialogOpen(false)}><Icon name="close" /></button></div>
            <p>Keep related conversations together. Projects are private to this browser account.</p>
            <label className="field-label">Project name<input autoFocus value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="e.g. Product research" maxLength={60} required /></label>
            <div className="modal-actions"><button type="button" className="button button-quiet" onClick={() => setProjectDialogOpen(false)}>Cancel</button><button type="submit" className="button button-primary">Create project</button></div>
          </form>
        </div>
      )}
    </div>
  );
}

interface ComposerProps {
  prompt: string;
  setPrompt: (value: string) => void;
  attachments: Array<{ name: string; content: string }>;
  removeAttachment: (name: string) => void;
  onAttach: () => void;
  onFiles: (files: FileList | null) => void;
  onSubmit: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  busy: boolean;
  composer: { current: HTMLTextAreaElement | null };
  fileInput: { current: HTMLInputElement | null };
}

function Composer(props: ComposerProps) {
  const { prompt, setPrompt, attachments, removeAttachment, onAttach, onFiles, onSubmit, onKeyDown, busy, composer, fileInput } = props;
  return (
    <div className="composer-wrap">
      {attachments.length > 0 && <div className="attachment-chips">{attachments.map((file) => <span className="attachment-chip" key={file.name}><Icon name="file" size={14} />{file.name}<button aria-label={`Remove ${file.name}`} onClick={() => removeAttachment(file.name)}><Icon name="close" size={13} /></button></span>)}</div>}
      <div className="composer">
        <textarea ref={composer} value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={onKeyDown} placeholder="Message DGB…" rows={2} aria-label="Message DGB" />
        <div className="composer-toolbar">
          <div className="composer-tools">
            <button className="icon-button" aria-label="Attach a file" title="Attach a text file" onClick={onAttach}><Icon name="attachment" size={17} /></button>
            <input ref={fileInput} className="visually-hidden" type="file" multiple accept=".txt,.md,.csv,.json,.log,text/plain,text/markdown,application/json" onChange={(event) => onFiles(event.target.files)} />
            <span className="composer-hint">DGB can make mistakes. Check important information.</span>
          </div>
          <button className={`send-button ${busy ? "send-stop" : ""}`} disabled={busy ? false : !prompt.trim()} aria-label={busy ? "Stop response" : "Send message"} onClick={busy ? () => window.dispatchEvent(new Event("dgb:stop")) : onSubmit}>
            <Icon name={busy ? "close" : "send"} size={16} />
          </button>
        </div>
      </div>
      <span className="composer-shortcut">Enter to send · Shift + Enter for a new line</span>
    </div>
  );
}
