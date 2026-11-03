import { useMemo, useSyncExternalStore } from "react";
import { AuthUser } from "@/types";

const TOKEN_KEY = "access_token";
const USER_KEY = "auth_user";
const AUTH_CHANGE_EVENT = "auth-change";

// localStorage only exists in the browser (safe during SSR / hydration).
function getStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null; // storage blocked (e.g. privacy mode)
  }
}

// Same-tab writes do not fire the native "storage" event, so emit our own.
function notifyAuthChange(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(AUTH_CHANGE_EVENT));
}

function parseUser(raw: string | null): AuthUser | null {
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && typeof (parsed as AuthUser).email === "string") {
      return parsed as AuthUser;
    }
  } catch {
    /* corrupted value */
  }
  return null;
}

// ---- access token ----
export function getToken(): string | null {
  return getStorage()?.getItem(TOKEN_KEY) ?? null;
}

export function setToken(token: string): void {
  getStorage()?.setItem(TOKEN_KEY, token);
  notifyAuthChange();
}

export function clearToken(): void {
  getStorage()?.removeItem(TOKEN_KEY);
  notifyAuthChange();
}

// ---- stored user ----
export function getStoredUser(): AuthUser | null {
  return parseUser(getStorage()?.getItem(USER_KEY) ?? null);
}

export function setStoredUser(user: AuthUser): void {
  getStorage()?.setItem(USER_KEY, JSON.stringify(user));
  notifyAuthChange();
}

export function clearStoredUser(): void {
  getStorage()?.removeItem(USER_KEY);
  notifyAuthChange();
}

export function clearAuth(): void {
  clearToken();
  clearStoredUser();
}

// ---- reactive hooks ----
// useSyncExternalStore: server snapshot is null, so there is no hydration mismatch
// and no setState-in-effect is needed to read localStorage.
function subscribeAuth(callback: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(AUTH_CHANGE_EVENT, callback);
  window.addEventListener("storage", callback); // login / logout in another tab
  return () => {
    window.removeEventListener(AUTH_CHANGE_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

function hasTokenSnapshot(): boolean {
  return getToken() !== null;
}

function rawUserSnapshot(): string | null {
  return getStorage()?.getItem(USER_KEY) ?? null;
}

function unknownSnapshot(): null {
  return null;
}

/** true / false in the browser; null while unknown (server render + hydration). */
export function useHasToken(): boolean | null {
  return useSyncExternalStore<boolean | null>(subscribeAuth, hasTokenSnapshot, unknownSnapshot);
}

/** Logged-in user from localStorage; updates on login / logout. */
export function useStoredUser(): AuthUser | null {
  const raw = useSyncExternalStore<string | null>(subscribeAuth, rawUserSnapshot, unknownSnapshot);
  return useMemo(() => parseUser(raw), [raw]);
}
