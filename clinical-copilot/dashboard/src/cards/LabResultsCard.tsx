import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; }

/**
 * Lab Results — `Observation?category=laboratory`. Mostly empty in
 * practice today; lab uploads currently land as a SOAP-note objective
 * line (see writer.write_lab_encounter_with_results) and don't yet
 * write FHIR Observation rows. The empty-state copy below explains
 * that to anyone wondering why the tab is silent after an upload.
 */
export function LabResultsCard({ pid }: Props) {
  const q = useQuery({
    queryKey: ['lab-results', pid],
    queryFn: () => api.labResults(pid),
  });

  return (
    <div className="card">
      <h2>
        Lab Results
        <a className="out-link" href={classicLinks.labResults(pid)} target="_blank" rel="noreferrer">
          Open in classic ↗
        </a>
      </h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load lab results (HTTP {q.error.status}).</div>}
      {q.data && q.data.items.length === 0 && (
        <div className="empty">No lab results on file.</div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul>
          {q.data.items.map((r) => (
            <li key={r.id}>
              <span className="primary">{r.test_name}</span>
              <span className="primary" style={{ marginLeft: 'auto' }}>
                {r.value != null ? r.value : (r.value_string || '—')}
                {r.unit ? ` ${r.unit}` : ''}
              </span>
              {r.abnormal_flag && (
                <span className={`pill ${
                  r.abnormal_flag === 'C' || r.abnormal_flag === 'HH' || r.abnormal_flag === 'LL'
                    ? 'pill-danger'
                    : (r.abnormal_flag === 'H' || r.abnormal_flag === 'L' || r.abnormal_flag === 'A')
                      ? 'pill-warn'
                      : 'pill-muted'
                }`}>{r.abnormal_flag}</span>
              )}
              {r.reference_range_text && (
                <span className="secondary" style={{ flexBasis: '100%' }}>ref {r.reference_range_text}</span>
              )}
              {r.effective && <span className="secondary">{r.effective.slice(0, 10)}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
