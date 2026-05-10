import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; numericPid: string | null; }

/**
 * Format the user's requested 3-line layout:
 *   [Physician name] - [Physician Specialty]
 *   [Physician phone numbers]
 *   [Physician address]
 *
 * Practice + NPI render as additional supporting lines when present.
 * Missing fields drop quietly — we never render placeholder dashes for
 * "no phone" because that just adds visual noise on a sparse row.
 */
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
        <div className="empty">
          No care team members on file. Care-team entries are populated
          from referring-physician contact blocks on uploaded referral
          letters.
        </div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul className="care-team-list">
          {q.data.items.map((m) => {
            const headLeft = m.name || '(unnamed)';
            const headRight = m.specialty;
            return (
              <li key={m.id} className="care-team-entry">
                <div className="ct-headline">
                  <span className="ct-name">{headLeft}</span>
                  {headRight && (
                    <>
                      <span className="ct-sep"> — </span>
                      <span className="ct-specialty">{headRight}</span>
                    </>
                  )}
                </div>
                {m.phone && <div className="ct-line">{m.phone}</div>}
                {m.address && <div className="ct-line">{m.address}</div>}
                {m.practice && <div className="ct-line ct-muted">{m.practice}</div>}
                {m.npi && <div className="ct-line ct-muted">NPI: {m.npi}</div>}
                {m.source === 'extracted' && (
                  <div className="ct-line ct-muted">From uploaded referral letter</div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
