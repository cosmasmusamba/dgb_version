import { FormEvent, useState } from "react";
import { ApiError, enterUiPreview, login } from "../api";
import type { User } from "../types";
import Icon from "./Icon";

interface LoginProps {
  onLogin: (user: User) => void;
}

export default function Login({ onLogin }: LoginProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      onLogin(await login(username.trim(), password));
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "Could not connect to the DGB service.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login-shell">
      <div className="login-backdrop login-backdrop-one" />
      <div className="login-backdrop login-backdrop-two" />
      <section className="login-card">
        <div className="brand-lockup login-brand">
          <span className="brand-mark">d</span>
          <span>DGB <span className="brand-sub">/ workspace</span></span>
        </div>
        <div className="login-heading">
          <span className="eyebrow"><Icon name="spark" size={15} /> A quieter place to think</span>
          <h1>Welcome back.</h1>
          <p>Sign in with the account provisioned for your DGB workspace.</p>
        </div>
        <form className="login-form" onSubmit={submit}>
          <label>
            Username
            <input autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required />
          </label>
          <label>
            Password
            <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </label>
          {error && <div className="form-error" role="alert"><Icon name="warning" size={16} /> {error}</div>}
          <button className="button button-primary login-submit" type="submit" disabled={busy}>
            {busy ? "Signing in…" : "Continue"} <Icon name="arrow" size={16} />
          </button>
        </form>
        {import.meta.env.DEV && <div className="preview-login">
          <div className="preview-divider"><span /> development preview · fixtures only <span /></div>
          <div className="preview-login-actions">
            <button type="button" className="button button-outline" onClick={() => onLogin(enterUiPreview("user"))}><Icon name="message" size={15} /> Preview user workspace</button>
            <button type="button" className="button button-quiet" onClick={() => onLogin(enterUiPreview("admin"))}><Icon name="admin" size={15} /> Preview admin workspace</button>
          </div>
          <p>Preview identities never authenticate to the API. Administrative actions and data are simulated locally.</p>
        </div>}
        <p className="login-note">Accounts are provisioned by your DGB administrator. Public registration is disabled.</p>
      </section>
      <footer className="login-footer"><span>Designed for considered work.</span><span>DGB AI · Private workspace</span></footer>
    </main>
  );
}
