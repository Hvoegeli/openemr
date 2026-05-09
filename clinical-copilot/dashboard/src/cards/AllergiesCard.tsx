import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; numericPid: string | null; }

export function AllergiesCard({ pid, numericPid }: Props) {
  const q = useQuery({
    queryKey: ['allergies', pid],
    queryFn: () => api.allergies(pid),
  });
  const classicUrl = classicLinks.allergies(numericPid);

  return (
    <div className="card">
      <h2>
        Allergies
        {classicUrl && (
          <a className="out-link" href={classicUrl} target="_blank" rel="noreferrer">
            Open in classic ↗
          </a>
        )}
      </h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load allergies (HTTP {q.error.status}).</div>}
      {q.data && q.data.items.length === 0 && (
        <div className="empty">No allergies on file.</div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul>
          {q.data.items.map((a) => (
            <li key={a.id}>
              <span className="primary">{a.display}</span>
              {a.criticality === 'high' && <span className="pill pill-danger">High</span>}
              {a.criticality === 'low' && <span className="pill pill-warn">Low</span>}
              {a.clinical_status && <span className="secondary">{a.clinical_status}</span>}
              {a.recorded_date && <span className="secondary">recorded {a.recorded_date}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
