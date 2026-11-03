// Small date / number formatting helpers shared by the session bar, visits panel, share page and dashboard.

const pad = (n: number) => String(n).padStart(2, "0");

/** "YYYY-MM-DD HH:mm" in local time; "" for an unparsable value. */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** "YYYY-MM-DD" in local time; a bare date string ("2026-11-03") is returned unchanged. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) return iso;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** Today as "YYYY-MM-DD" (local time), for date inputs. */
export function todayIsoDate(): string {
  const d = new Date();
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** 0.734 → "73.4%" */
export function formatPercent(fraction: number | null | undefined, digits = 1): string {
  if (fraction == null || !Number.isFinite(fraction)) return "–";
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** 0.01234 → "$0.0123"; null → "–" */
export function formatUsd(usd: number | null | undefined): string {
  if (usd == null || !Number.isFinite(usd)) return "–";
  return `$${Number(usd.toFixed(4))}`;
}

/** 1234 → "1.2s"; 850 → "0.9s"; null → "–" */
export function formatLatency(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return "–";
  return `${(ms / 1000).toFixed(1)}s`;
}

/** 1234 → "1 KB"; 15_728_640 → "15.0 MB" (file pickers / upload summaries). */
export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "–";
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

/** NIfTI volume / label map by file name or URL path (".nii" / ".nii.gz", case-insensitive). */
export function isNiftiPath(name: string): boolean {
  const base = name.split(/[?#]/)[0].toLowerCase();
  return base.endsWith(".nii") || base.endsWith(".nii.gz");
}
