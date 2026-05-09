/**
 * Dashboard root.
 *
 * URL contract: `/dashboard?pid=<patient-uuid>`. Without a `pid` we
 * render a small picker that nudges users back to Copilot's chat
 * surface (which is the canonical patient-selector). Production keeps
 * the dashboard tight by NOT reimplementing patient search — that's
 * the Copilot's job; the dashboard renders one patient at a time.
 *
 * Auth contract: every API call requires the Copilot session cookie.
 * On 401 we surface a "Sign in" CTA that points at the OAuth2 entry
 * (`/oauth/login?next=/dashboard?pid=...`) so the round-trip lands
 * the user back on this same view post-auth.
 *
 * Layout: patient banner on top, tab bar under it (one tab per
 * clinical card), and a single content panel below the bar that
 * renders the active tab's card. Tabs replace the old grid layout
 * because clinicians scan one section at a time — having all six
 * cards visible at once was extra cognitive load with no upside.
 */

import { useState, ComponentType } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ApiError, api } from './api';
import { PatientBanner } from './cards/PatientBanner';
import { AllergiesCard } from './cards/AllergiesCard';
import { ConditionsCard } from './cards/ConditionsCard';
import { MedicationsCard } from './cards/MedicationsCard';
import { PrescriptionsCard } from './cards/PrescriptionsCard';
import { LabResultsCard } from './cards/LabResultsCard';
import { OrdersCard } from './cards/OrdersCard';
import { CareTeamCard } from './cards/CareTeamCard';
import { VitalsCard } from './cards/VitalsCard';

interface TabDef {
  id: string;
  label: string;
  Component: ComponentType<{ pid: string }>;
}

// Order: rubric-required cards first (Allergies, Problem List,
// Medications, Prescriptions, Care Team), then the chosen optional
// section (Vitals), then the empty-state-by-default surfaces (Lab
// Results, Orders) which fill in as Co-Pilot writes data.
const TABS: TabDef[] = [
  { id: 'allergies',     label: 'Allergies',     Component: AllergiesCard },
  { id: 'problems',      label: 'Problem List',  Component: ConditionsCard },
  { id: 'medications',   label: 'Medications',   Component: MedicationsCard },
  { id: 'prescriptions', label: 'Prescriptions', Component: PrescriptionsCard },
  { id: 'lab-results',   label: 'Lab Results',   Component: LabResultsCard },
  { id: 'orders',        label: 'Orders',        Component: OrdersCard },
  { id: 'vitals',        label: 'Vitals',        Component: VitalsCard },
  { id: 'care-team',     label: 'Care Team',     Component: CareTeamCard },
];

function getPidFromUrl(): string | null {
  const params = new URLSearchParams(window.location.search);
  const pid = params.get('pid');
  return pid && pid.trim() ? pid.trim() : null;
}

function loginHref(): string {
  const next = `/dashboard${window.location.search}`;
  return `/oauth/login?next=${encodeURIComponent(next)}`;
}

export function App() {
  const pid = getPidFromUrl();
  const [activeTabId, setActiveTabId] = useState<string>(TABS[0].id);

  // Drive the auth check off the patient header query — it's the first
  // thing every other card depends on, so a 401 here short-circuits the
  // page into the login prompt without flashing partial cards.
  const headerQuery = useQuery({
    queryKey: ['patientHeader', pid],
    queryFn: () => api.patientHeader(pid!),
    enabled: !!pid,
  });

  if (!pid) {
    return (
      <div className="app-shell">
        <div className="login-prompt">
          <h1>Modern Dashboard</h1>
          <p>This page renders one patient at a time. Open it from the Co-Pilot
             chat (a "View on Modern Dashboard" link will appear after you
             reference a patient) or pass <code>?pid=&lt;patient-uuid&gt;</code>
             in the URL.</p>
        </div>
      </div>
    );
  }

  if (headerQuery.error instanceof ApiError && headerQuery.error.status === 401) {
    return (
      <div className="app-shell">
        <div className="login-prompt">
          <h1>Sign in to view this patient</h1>
          <p>You need to authenticate with OpenEMR before the dashboard can
             load chart data.</p>
          <a href={loginHref()}>Sign in via OpenEMR</a>
        </div>
      </div>
    );
  }

  const ActiveCard = (TABS.find(t => t.id === activeTabId) || TABS[0]).Component;

  return (
    <div className="app-shell">
      <PatientBanner pid={pid} headerQuery={headerQuery} />
      <nav className="tab-bar" role="tablist" aria-label="Patient sections">
        {TABS.map(t => {
          const isActive = t.id === activeTabId;
          return (
            <button
              key={t.id}
              role="tab"
              aria-selected={isActive}
              className={`tab-btn${isActive ? ' active' : ''}`}
              onClick={() => setActiveTabId(t.id)}
              type="button"
            >
              {t.label}
            </button>
          );
        })}
      </nav>
      <section className="tab-panel" role="tabpanel">
        <ActiveCard pid={pid} />
      </section>
    </div>
  );
}
