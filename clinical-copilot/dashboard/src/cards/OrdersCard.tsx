import { useQuery } from '@tanstack/react-query';
import { ApiError, api, classicLinks } from '../api';

interface Props { pid: string; }

/**
 * Orders — `ServiceRequest`. Lab orders, imaging orders, referrals,
 * and procedure orders placed for this patient. Empty state by
 * default until OpenEMR surfaces them via FHIR (or until Co-Pilot's
 * clinical-note flow learns to write ServiceRequest entries for
 * "follow-up labs" / "imaging requested" lines from prior shifts).
 */
export function OrdersCard({ pid }: Props) {
  const q = useQuery({
    queryKey: ['orders', pid],
    queryFn: () => api.orders(pid),
  });

  return (
    <div className="card">
      <h2>
        Orders
        <a className="out-link" href={classicLinks.orders(pid)} target="_blank" rel="noreferrer">
          Open in classic ↗
        </a>
      </h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load orders (HTTP {q.error.status}).</div>}
      {q.data && q.data.items.length === 0 && (
        <div className="empty">No orders on file.</div>
      )}
      {q.data && q.data.items.length > 0 && (
        <ul>
          {q.data.items.map((o) => (
            <li key={o.id}>
              <span className="primary">{o.display}</span>
              {o.priority === 'urgent' || o.priority === 'asap' || o.priority === 'stat' ? (
                <span className="pill pill-danger">{o.priority}</span>
              ) : null}
              {o.status && <span className="secondary">{o.status}</span>}
              {o.category && <span className="secondary">{o.category}</span>}
              {o.authored && <span className="secondary">authored {o.authored.slice(0, 10)}</span>}
              {o.requester && (
                <span className="secondary" style={{ flexBasis: '100%' }}>by {o.requester}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
