import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; numericPid: string | null; }

export function CareTeamCard({ pid, numericPid }: Props) {
  const q = useQuery({
    queryKey: ['careTeam', pid],
    queryFn: () => api.careTeam(pid),
  });
  const classicUrl = classicLinks.careTeam(numericPid);

  return (
    <div className="card">
      <h2>
        Care Team
        {classicUrl && (
          <a className="out-link" href={classicUrl} target="_blank" rel="noreferrer">
            Open in classic ↗
          </a>
        )}
      </h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load care team (HTTP {q.error.status}).</div>}
      {q.data && q.data.items.length === 0 && (
        <div className="empty">No care team members on file.</div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul>
          {q.data.items.map((m) => (
            <li key={m.id}>
              <span className="primary">{m.name || '(unnamed)'}</span>
              {m.role && <span className="secondary">{m.role}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
