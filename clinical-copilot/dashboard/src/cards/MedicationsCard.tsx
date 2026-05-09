import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; }

export function MedicationsCard({ pid }: Props) {
  const q = useQuery({
    queryKey: ['medications', pid],
    queryFn: () => api.medications(pid),
  });

  return (
    <div className="card">
      <h2>
        Medications
        <a className="out-link" href={classicLinks.medications(pid)} target="_blank" rel="noreferrer">
          Open in classic ↗
        </a>
      </h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load medications (HTTP {q.error.status}).</div>}
      {q.data && q.data.items.length === 0 && (
        <div className="empty">No medications on file.</div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul>
          {q.data.items.map((m) => (
            <li key={m.id}>
              <span className="primary">{m.display}</span>
              {m.status && <span className="secondary">{m.status}</span>}
              {m.authored_date && <span className="secondary">authored {m.authored_date}</span>}
              {m.dosage_text && <span className="secondary" style={{ flexBasis: '100%' }}>{m.dosage_text}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
