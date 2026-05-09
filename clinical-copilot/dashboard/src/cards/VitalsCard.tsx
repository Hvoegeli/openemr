import { useQuery } from '@tanstack/react-query';
import { ApiError, api, VitalReading, VitalSeries } from '../api';

interface Props { pid: string; numericPid: string | null; }

/**
 * Lightweight inline sparkline. No charting library — for a 6-12 point
 * vitals series the SVG path generation is ~20 lines and saves us
 * 80kB of dependency. Goal-band overlay is per-LOINC. Datapoint
 * values are labeled inline (#2 from review feedback) so the user
 * doesn't have to hover or guess.
 */
function Sparkline({
  readings,
  low,
  high,
  stroke = '#1f6feb',
  showLabels = false,
}: {
  readings: VitalReading[];
  low?: number;
  high?: number;
  stroke?: string;
  showLabels?: boolean;
}) {
  const numeric = readings.filter(r => r.value != null).map(r => r.value as number);
  if (numeric.length < 2) return null;

  const W = 220;
  const H = 60;
  const PAD = 8;
  const TEXT_PAD = 10;
  const minV = Math.min(...numeric, low ?? Infinity);
  const maxV = Math.max(...numeric, high ?? -Infinity);
  const span = maxV - minV || 1;
  const xStep = (W - PAD * 2) / (numeric.length - 1);
  const coords = numeric.map((v, i) => {
    const x = PAD + i * xStep;
    const y = PAD + (1 - (v - minV) / span) * (H - PAD * 2);
    return { x, y, v };
  });

  const bandRect =
    low != null && high != null && low < high
      ? (() => {
          const yLow = PAD + (1 - (high - minV) / span) * (H - PAD * 2);
          const yHigh = PAD + (1 - (low - minV) / span) * (H - PAD * 2);
          const top = Math.min(yLow, yHigh);
          const bottom = Math.max(yLow, yHigh);
          return <rect x={PAD} y={top} width={W - PAD * 2} height={Math.max(0, bottom - top)} fill="#e1f3e6" />;
        })()
      : null;

  return (
    <svg width={W} height={H} role="img" aria-label="vitals trend" style={{ overflow: 'visible' }}>
      {bandRect}
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth={1.5}
        points={coords.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}
      />
      {coords.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={2.5} fill={stroke} />
      ))}
      {showLabels && coords.map((p, i) => {
        // Anchor labels above the point for the top half of the chart,
        // below for the bottom half — keeps them clear of the line.
        const above = p.y > H / 2;
        const dy = above ? -TEXT_PAD : TEXT_PAD + 4;
        // Only render every Nth label when the series is dense, to
        // avoid overlap. Always render the first and last.
        const stride = coords.length > 8 ? Math.ceil(coords.length / 6) : 1;
        if (i !== 0 && i !== coords.length - 1 && i % stride !== 0) return null;
        return (
          <text
            key={`l${i}`}
            x={p.x}
            y={p.y + dy}
            fontSize={9}
            fill="#5a6376"
            textAnchor="middle"
          >
            {Number.isInteger(p.v) ? p.v : p.v.toFixed(1)}
          </text>
        );
      })}
    </svg>
  );
}

// Per-LOINC goal bands. Conservative — only display bands we have a
// guideline source for. Systolic BP ranges per JNC8 / ACC/AHA 2017
// (general adult target <130). Resting HR normal range 60-100.
const GOAL_BANDS: Record<string, { low: number; high: number; label: string } | undefined> = {
  '8480-6': { low: 90, high: 130, label: 'JNC8 systolic target <130' },
  '8462-4': { low: 60, high: 80,  label: 'JNC8 diastolic target <80' },
  '8867-4': { low: 60, high: 100, label: 'Resting HR normal 60-100' },
};

const SYSTOLIC_LOINC = '8480-6';
const DIASTOLIC_LOINC = '8462-4';

/**
 * Combined Blood Pressure panel — pairs systolic + diastolic by
 * timestamp and renders a single "Blood Pressure" header with the
 * latest "120/80 mmHg" reading. Both sparklines are stacked underneath,
 * with the systolic line in blue and diastolic in green so the
 * relationship is visible at a glance.
 */
function BloodPressurePanel({
  systolic,
  diastolic,
}: {
  systolic: VitalSeries;
  diastolic: VitalSeries;
}) {
  // Pair readings by `effectiveDateTime` so the "latest" reading shows
  // both numbers from the same encounter, not a sys-from-Tuesday +
  // dia-from-last-week mismatch.
  const dyn = new Map<string, { sys?: number; dia?: number }>();
  for (const r of systolic.readings) {
    if (r.value != null && r.effective) {
      const slot = dyn.get(r.effective) || {};
      slot.sys = r.value;
      dyn.set(r.effective, slot);
    }
  }
  for (const r of diastolic.readings) {
    if (r.value != null && r.effective) {
      const slot = dyn.get(r.effective) || {};
      slot.dia = r.value;
      dyn.set(r.effective, slot);
    }
  }
  const paired = [...dyn.entries()]
    .filter(([, v]) => v.sys != null && v.dia != null)
    .sort(([a], [b]) => a.localeCompare(b));
  const latest = paired.length > 0 ? paired[paired.length - 1][1] : null;
  const sysBand = GOAL_BANDS[SYSTOLIC_LOINC];
  const diaBand = GOAL_BANDS[DIASTOLIC_LOINC];

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <strong>Blood Pressure</strong>
        {latest && (
          <span style={{ marginLeft: 'auto' }} className="secondary">
            latest: {latest.sys}/{latest.dia} mmHg
          </span>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <Sparkline readings={systolic.readings} low={sysBand?.low} high={sysBand?.high} stroke="#1f6feb" showLabels />
        <Sparkline readings={diastolic.readings} low={diaBand?.low} high={diaBand?.high} stroke="#1e7e34" showLabels />
      </div>
      <div className="secondary" style={{ fontSize: 11 }}>
        Systolic (blue) target &lt;130 · Diastolic (green) target &lt;80
      </div>
    </div>
  );
}

export function VitalsCard({ pid }: Props) { // numericPid intentionally unused — Vitals has no classic-OpenEMR out-link target
  const q = useQuery({
    queryKey: ['vitals', pid],
    queryFn: () => api.vitals(pid),
  });

  // Pull systolic/diastolic out of the series list and render them as
  // one combined Blood Pressure panel. Everything else falls through
  // to the generic per-series panel below.
  const series = q.data?.series ?? [];
  const sys = series.find(s => s.loinc === SYSTOLIC_LOINC);
  const dia = series.find(s => s.loinc === DIASTOLIC_LOINC);
  const others = series.filter(s => s.loinc !== SYSTOLIC_LOINC && s.loinc !== DIASTOLIC_LOINC);

  return (
    <div className="card">
      <h2>Vitals</h2>
      {q.isLoading && <div className="loading">Loading…</div>}
      {q.error instanceof ApiError && <div className="error">Failed to load vitals (HTTP {q.error.status}).</div>}
      {q.data && series.length === 0 && (
        <div className="empty">No vital signs recorded.</div>
      )}
      {q.data && series.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16 }}>
          {sys && dia && <BloodPressurePanel systolic={sys} diastolic={dia} />}
          {others.map((s) => {
            const band = GOAL_BANDS[s.loinc];
            const last = [...s.readings].reverse().find(r => r.value != null);
            return (
              <div key={s.loinc}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                  <strong>{s.display}</strong>
                  {last && (
                    <span style={{ marginLeft: 'auto' }} className="secondary">
                      latest: {last.value}{s.unit ? ` ${s.unit}` : ''}
                    </span>
                  )}
                </div>
                <Sparkline readings={s.readings} low={band?.low} high={band?.high} showLabels />
                {band && <div className="secondary" style={{ fontSize: 11 }}>{band.label}</div>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
