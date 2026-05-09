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

export const classicLinks = {
  patientSummary: (pid: string) =>
    `${classicBase()}/interface/patient_file/summary/demographics.php?site=default&set_pid=${encodeURIComponent(pid)}`,
  allergies: (pid: string) =>
    `${classicBase()}/interface/patient_file/summary/stats_full.php?site=default&set_pid=${encodeURIComponent(pid)}&category=allergy`,
  problems: (pid: string) =>
    `${classicBase()}/interface/patient_file/summary/stats_full.php?site=default&set_pid=${encodeURIComponent(pid)}&category=medical_problem`,
  medications: (pid: string) =>
    `${classicBase()}/interface/patient_file/summary/stats_full.php?site=default&set_pid=${encodeURIComponent(pid)}&category=medication`,
  prescriptions: (pid: string) =>
    `${classicBase()}/interface/patient_file/summary/stats_full.php?site=default&set_pid=${encodeURIComponent(pid)}&category=prescription`,
  labResults: (pid: string) =>
    `${classicBase()}/interface/orders/orders_results.php?site=default&pid=${encodeURIComponent(pid)}`,
  orders: (pid: string) =>
    `${classicBase()}/interface/orders/types.php?site=default&pid=${encodeURIComponent(pid)}`,
  careTeam: (pid: string) =>
    `${classicBase()}/interface/patient_file/summary/demographics.php?site=default&set_pid=${encodeURIComponent(pid)}`,
};
