import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; }

/**
 * Prescriptions card — administrative-order view of MedicationRequest.
 *
 * Required by the dashboard rubric as a sibling to the Medications
 * card. Same FHIR resource as Medications, but the fields surfaced
 * here are the ones a prescriber/admin cares about: who wrote it,
 * when, refills, dispense quantity. Sorted most-recent first.
 */
export function PrescriptionsCard({ pid }: Props) {
  const q = useQuery({
    queryKey: ['prescriptions', pid],
    queryFn: () => api.prescriptions(pid),
  });

  return (
    <div className="card">
      <h2>
        Prescriptions
        <a className="out-link" href={classicLinks.prescriptions(pid)} target="_blank" rel="noreferrer">
          Open in classic ↗
        </a>
      </h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load prescriptions (HTTP {q.error.status}).</div>}
      {q.data && q.data.items.length === 0 && (
        <div className="empty">No prescription orders on file.</div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul>
          {q.data.items.map((p) => (
            <li key={p.id}>
              <span className="primary">{p.display}</span>
              {p.status && <span className="secondary">{p.status}</span>}
              {p.authored_date && <span className="secondary">written {p.authored_date}</span>}
              {p.prescriber && (
                <span className="secondary" style={{ flexBasis: '100%' }}>by {p.prescriber}</span>
              )}
              {(p.refills != null || p.quantity != null) && (
                <span className="secondary" style={{ flexBasis: '100%' }}>
                  {p.quantity != null ? `qty ${p.quantity}` : ''}
                  {p.quantity != null && p.refills != null ? ' · ' : ''}
                  {p.refills != null ? `${p.refills} refills` : ''}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
