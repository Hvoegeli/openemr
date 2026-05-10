/**
 * Copilot API client — narrow surface for the dashboard.
 *
 * All calls go to same-origin `/api/dashboard/...`. The session cookie
 * (set by Copilot's OAuth2 callback) authenticates every request — no
 * tokens flow through the browser. CORS isn't a concern because the
 * dashboard is served from the same origin as the Copilot API.
 */

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    credentials: 'include',
    headers: { Accept: 'application/json' },
  });
  if (res.status === 401) {
    throw new ApiError(401, 'Not authenticated');
  }
  if (!res.ok) {
    throw new ApiError(res.status, `HTTP ${res.status} on ${path}`);
  }
  return res.json() as Promise<T>;
}

// Type shapes are deliberately narrow — only the fields the cards
// actually render. Adding fields is a one-line change here, but
// rendering work that's not in the demo loop wastes time.

export interface PatientHeader {
  id: string;
  full_name: string;
  given: string[];
  family: string;
  birth_date: string | null;
  age: number | null;
  gender: string | null;
  mrn: string | null;
  active: boolean;
  primary_phone: string | null;
  /**
   * OpenEMR's internal numeric pid for this patient. Required for the
   * "Open in classic OpenEMR" out-link URLs because classic's
   * `set_pid` parameter calls `intval()` on the value and rejects
   * non-numeric input — passing the FHIR UUID would land users on
   * patient #0. `null` when the identifier shape varies and we can't
   * resolve it; the frontend disables the out-link in that case.
   */
  numeric_pid: string | null;
}

export interface AllergyEntry {
  id: string;
  display: string;
  criticality: 'low' | 'high' | 'unable-to-assess' | null;
  clinical_status: string | null;
  recorded_date: string | null;
}

export interface ConditionEntry {
  id: string;
  display: string;
  clinical_status: string | null;
  onset_date: string | null;
}

export interface MedicationEntry {
  id: string;
  display: string;
  status: string | null;
  authored_date: string | null;
  dosage_text: string | null;
}

export interface PrescriptionEntry {
  id: string;
  display: string;
  status: string | null;
  intent: string | null;
  authored_date: string | null;
  prescriber: string | null;
  refills: number | null;
  quantity: number | null;
}

export interface CareTeamEntry {
  id: string;
  name: string | null;
  role: string | null;
}

export interface VitalReading {
  effective: string;
  value: number | null;
  unit: string | null;
}
export interface VitalSeries {
  loinc: string;
  display: string;
  unit: string | null;
  readings: VitalReading[];
}

export interface LabResultEntry {
  id: string;
  loinc: string | null;
  test_name: string;
  value: number | null;
  value_string: string | null;
  unit: string | null;
  reference_range_text: string | null;
  abnormal_flag: 'H' | 'L' | 'N' | 'C' | 'HH' | 'LL' | 'A' | null;
  effective: string | null;
  status: string | null;
}

export interface OrderEntry {
  id: string;
  display: string;
  category: string | null;
  status: string | null;
  intent: string | null;
  priority: string | null;
  authored: string | null;
  requester: string | null;
}

export const api = {
  patientHeader: (pid: string) =>
    getJSON<PatientHeader>(`/api/dashboard/patient/${encodeURIComponent(pid)}/header`),

  allergies: (pid: string) =>
    getJSON<{ items: AllergyEntry[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/allergies`),

  conditions: (pid: string) =>
    getJSON<{ items: ConditionEntry[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/conditions`),

  medications: (pid: string) =>
    getJSON<{ items: MedicationEntry[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/medications`),

  prescriptions: (pid: string) =>
    getJSON<{ items: PrescriptionEntry[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/prescriptions`),

  careTeam: (pid: string) =>
    getJSON<{ items: CareTeamEntry[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/care-team`),

  vitals: (pid: string) =>
    getJSON<{ series: VitalSeries[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/vitals`),

  labResults: (pid: string) =>
    getJSON<{ items: LabResultEntry[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/lab-results`),

  orders: (pid: string) =>
    getJSON<{ items: OrderEntry[] }>(`/api/dashboard/patient/${encodeURIComponent(pid)}/orders`),
};

/**
 * URL helpers for "Open in classic OpenEMR" out-links.
 *
 * Classic OpenEMR is at the same origin as Copilot only when proxied
 * (Hetzner cloudflared puts both behind one tunnel). For local dev the
 * default points at https://localhost:9300; the app shell reads the
 * actual base from a meta tag injected by the dashboard's PHP/HTML
 * shell so deploys can override without rebuilding the bundle.
 *
 * Every URL carries `site=default` so OpenEMR's `interface/globals.php`
 * site-detection passes even when the user lands on the classic page
 * cold (without an existing OpenEMR session). Without it the page
 * returns "Site ID is missing from session data!" — same bug we fixed
 * on the modern_dashboard.php entry point.
 */
function classicBase(): string {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="openemr-classic-base"]');
  return meta?.content || 'https://localhost:9300';
}

/**
 * Build a classic-OpenEMR URL, or `null` when we can't.
 *
 * `numericPid` MUST be the OpenEMR-internal numeric pid (an integer
 * string), not the FHIR Patient UUID. Classic's `set_pid` parameter
 * calls `intval()` on the value (see `src/Common/Session/PatientSessionUtil.php`),
 * so a UUID becomes `0` and the page lands on the wrong (or no)
 * patient. When the resolver could not produce a numeric pid (legacy
 * deployments where the identifier shape differs), every link below
 * returns `null` — the cards then disable their out-link button rather
 * than emit a URL that would silently load the wrong chart.
 *
 * All URLs route through `interface/main/openpatient.php` instead of
 * landing on the requested deep-link path directly. openpatient.php:
 *   - if the user is logged in to classic OpenEMR, sets the session
 *     pid and redirects to demographics.php — the chart loads with
 *     the patient already selected;
 *   - if not logged in, redirects to login.php?patientID=N, which
 *     puts patientID on the login form as a hidden input;
 *     main_screen.php (line ~446) honors that on POST to auto-open
 *     the patient's chart tab post-login.
 *
 * Without this hop, the post-login redirect lands on OpenEMR's
 * generic home and the user has to search for the patient again.
 *
 * The dashboard cannot tell whether the user has a classic OpenEMR
 * session (cross-origin: Co-Pilot at one cloudflared subdomain,
 * classic OpenEMR at another, no shared cookies). openpatient.php
 * does the auth check server-side and branches accordingly, which
 * is the only correct place for that decision to live.
 */
type ClassicUrl = string | null;

function buildClassicUrl(numericPid: string | null): ClassicUrl {
  if (!numericPid) return null;
  const params = new URLSearchParams({ pid: numericPid });
  return `${classicBase()}/interface/main/openpatient.php?${params.toString()}`;
}

export const classicLinks = {
  patientSummary: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
  allergies: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
  problems: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
  medications: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
  prescriptions: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
  labResults: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
  orders: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
  careTeam: (numericPid: string | null): ClassicUrl => buildClassicUrl(numericPid),
};
