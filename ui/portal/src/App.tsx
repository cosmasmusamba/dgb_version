import { useEffect, useState } from "react";
import { clearSession, currentUser, getMe, isPreviewSession, updateUser } from "./api";
import AdminWorkspace from "./components/AdminWorkspace";
import ChatWorkspace from "./components/ChatWorkspace";
import Login from "./components/Login";
import { guestUser } from "./storage";
import type { User } from "./types";

type Theme = "light" | "dark";

function initialTheme(): Theme {
  const saved = localStorage.getItem("dgb.theme");
  if (saved === "dark" || saved === "light") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [user, setUser] = useState<User>(() => currentUser() ?? guestUser());
  const [checking, setChecking] = useState(() => Boolean(currentUser()));
  const [loginOpen, setLoginOpen] = useState(false);
  const [workspace, setWorkspace] = useState<"chat" | "admin">("chat");
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    let active = true;
    if (!user.isGuest && !isPreviewSession()) {
      void getMe().then((freshUser) => {
        if (!active) return;
        updateUser(freshUser);
        setUser(freshUser);
      }).catch(() => {
        if (!active) return;
        clearSession();
        setUser(guestUser());
      }).finally(() => {
        if (active) setChecking(false);
      });
    } else {
      setChecking(false);
    }
    return () => { active = false; };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("dgb.theme", theme);
  }, [theme]);

  function logout() {
    clearSession();
    setUser(guestUser());
    setLoginOpen(false);
    setWorkspace("chat");
  }

  function handleLogin(nextUser: User) {
    setUser(nextUser);
    setLoginOpen(false);
    if (isPreviewSession() && nextUser.role === "admin") setWorkspace("admin");
  }

  function continueAsGuest() {
    setUser(guestUser());
    setLoginOpen(false);
    setWorkspace("chat");
  }

  function enterAdmin() {
    if (user?.role === "admin") setWorkspace("admin");
  }

  if (checking) {
    return <main className="session-loading"><span className="loading-spinner" /><p>Checking your DGB session…</p></main>;
  }
  if (loginOpen && user.isGuest) return <Login onLogin={handleLogin} onContinueAsGuest={continueAsGuest} />;
  if (workspace === "admin" && user.role === "admin") {
    return <AdminWorkspace
      user={user}
      theme={theme}
      onToggleTheme={() => setTheme((value) => value === "light" ? "dark" : "light")}
      onBackToChat={() => setWorkspace("chat")}
      onLogout={logout}
    />;
  }
  return <ChatWorkspace
    user={user}
    theme={theme}
    onToggleTheme={() => setTheme((value) => value === "light" ? "dark" : "light")}
    onOpenAdmin={enterAdmin}
    onOpenLogin={() => setLoginOpen(true)}
    onLogout={logout}
  />;
}
