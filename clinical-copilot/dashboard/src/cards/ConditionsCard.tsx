import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; numericPid: string | null; }

export function ConditionsCard({ pid, numericPid }: Props) {
  const q = useQuery({
    queryKey: ['conditions', pid],
    queryFn: () => api.conditions(pid),
  });
  const classicUrl = classicLinks.problems(numericPid);

  return (
    <div className="card">
      <h2>
        Problem List
        {classicUrl && (
          <a className="out-link" href={classicUrl} target="_blank" rel="noreferrer">
            Open in classic ↗
          </a>
        )}
      </h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load problem list (HTTP {q.error.status}).</div>}
      {q.data && q.data.items.length === 0 && (
        <div className="empty">No active problems on file.</div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul>
          {q.data.items.map((c) => (
            <li key={c.id}>
              <span className="primary">{c.display}</span>
              {c.clinical_status && <span className="secondary">{c.clinical_status}</span>}
              {c.onset_date && <span className="secondary">onset {c.onset_date}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
