import { getTrustTier } from '../constants';

function formatCurrency(val) {
  if (val == null) return '—';
  return '$' + Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export default function StatStrip({ simpleView }) {
  if (!simpleView) return null;

  const channels = simpleView.channels || [];
  const totalContributionPct = channels.reduce(
    (sum, c) => sum + c.contribution_pct,
    0
  );
  const baselinePct = simpleView.baseline_pct ?? 0;
  const trustInfo = getTrustTier(simpleView.trust_tier);
  const recs = simpleView.recommendations || [];
  const actionCount = recs.length;

  return (
    <div className="stat-strip">
      <div className="stat-card">
        <div className="stat-label">Incremental Revenue</div>
        <div className="stat-value">{totalContributionPct.toFixed(1)}%</div>
        <div className="stat-delta neutral">of total revenue from ads</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Baseline Revenue</div>
        <div className="stat-value">{baselinePct.toFixed(1)}%</div>
        <div className="stat-delta neutral">trend + seasonality</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Model Trust</div>
        <div className="stat-value" style={{ color: trustInfo.color, fontSize: '1.1rem' }}>
          {trustInfo.shortLabel}
        </div>
        <div className="stat-delta neutral">
          Score: {(simpleView.trust_score * 100).toFixed(0)}%
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Actions Flagged</div>
        <div className="stat-value">{actionCount}</div>
        <div className="stat-delta neutral">
          {actionCount === 0 ? 'no actions needed' : 'recommendations below'}
        </div>
      </div>
    </div>
  );
}
