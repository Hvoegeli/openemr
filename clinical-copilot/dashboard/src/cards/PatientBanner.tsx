import type { MouseEvent as ReactMouseEvent } from 'react';
import { UseQueryResult } from '@tanstack/react-query';
import { PatientHeader, classicLinks } from '../api';

interface Props {
  pid: string;
  headerQuery: UseQueryResult<PatientHeader, unknown>;
}

/**
 * Co-Pilot home URL — the chat surface. Same origin as the dashboard,
 * so the existing session cookie carries the user back across without
 * a second login. Read from a meta tag so deploys can override the
 * base URL without rebuilding the bundle.
 */
function copilotHome(): string {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="copilot-base"]');
  return (meta?.content || '/').replace(/\/?$/, '/');
}

/**
 * "← Back to Co-Pilot" navigation.
 *
 * The dashboard is opened from the Co-Pilot chat surface via
 * `target="_blank"`, so this tab usually has a `window.opener`. The
 * desired UX is "the dashboard tab disappears and you're back in
 * your chat tab, on the patient card you started from."
 *
 * Three cases:
 *   1. Opener exists and is reachable — navigate the opener to
 *      `/?pid=<pid>` (Copilot's URL handler auto-selects the patient
 *      and switches to the Patient Card tab), focus the opener,
 *      close this dashboard tab.
 *   2. Opener exists but the user has navigated it away — same as
 *      above; the navigation will reload Copilot anyway.
 *   3. No opener (user opened the dashboard URL directly, or the
 *      browser stripped the opener handle) — fall back to
 *      `window.location.href = /?pid=<pid>` in this same tab.
 */
function returnToCopilot(pid: string, e: ReactMouseEvent<HTMLAnchorElement>): void {
  e.preventDefault();
  const target = `${copilotHome()}?pid=${encodeURIComponent(pid)}`;
  const opener = window.opener as Window | null;
  try {
    if (opener && !opener.closed) {
      opener.location.href = target;
      opener.focus();
      window.close();
      // If close was blocked (popup-blocker, security policy), the
      // tab is still here — fall through to in-tab navigation as a
      // belt-and-braces fallback.
      setTimeout(() => {
        if (!window.closed) window.location.href = target;
      }, 80);
      return;
    }
  } catch (_) {
    // Cross-origin opener access can throw; treat as no-opener.
  }
  window.location.href = target;
}

export function PatientBanner({ pid, headerQuery }: Props) {
  if (headerQuery.isLoading) {
    return <div className="patient-banner loading">Loading patient…</div>;
  }
  if (headerQuery.error || !headerQuery.data) {
    return <div className="patient-banner error">Failed to load patient header.</div>;
  }
  const p = headerQuery.data;
  const ageStr = p.age != null ? `${p.age}y` : '—';
  const dobStr = p.birth_date || '—';
  const genderStr = p.gender || '—';
  const mrnStr = p.mrn || '—';
  const statusPill = p.active
    ? <span className="pill pill-ok">Active</span>
    : <span className="pill pill-muted">Inactive</span>;

  return (
    <div className="patient-banner">
      <div>
        <h1>{p.full_name || `${p.given.join(' ')} ${p.family}`.trim() || '(unnamed patient)'}</h1>
        <div className="meta">
          {ageStr} · {genderStr} · DOB {dobStr} · MRN {mrnStr}{p.primary_phone ? ` · ${p.primary_phone}` : ''}
        </div>
      </div>
      <div>{statusPill}</div>
      <div className="banner-actions">
        <a
          className="out-link back-to-copilot"
          href={`${copilotHome()}?pid=${encodeURIComponent(pid)}`}
          onClick={(e) => returnToCopilot(pid, e)}
          title="Return to the Co-Pilot chat surface, on this patient's card"
        >
          ← Back to Co-Pilot
        </a>
        <a className="out-link" href={classicLinks.patientSummary(pid)} target="_blank" rel="noreferrer">
          Open in classic OpenEMR ↗
        </a>
      </div>
    </div>
  );
}
