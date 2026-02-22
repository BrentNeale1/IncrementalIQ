function statusClass(status) {
  if (status === 'sufficient') return 'good';
  if (status === 'caution') return 'warn';
  return 'bad';
}

export default function QualityReport({ quality }) {
  if (!quality) return null;

  return (
    <div className="card">
      <div className="card-title">Data Quality Report</div>
      <div className="quality-grid">
        <div className="quality-item">
          <div className="quality-item-label">History</div>
          <div className={`quality-item-value ${statusClass(quality.history_status)}`}>
            {quality.history_days} days
          </div>
        </div>
        <div className="quality-item">
          <div className="quality-item-label">Status</div>
          <div className={`quality-item-value ${statusClass(quality.history_status)}`}>
            {quality.history_status}
          </div>
        </div>
        <div className="quality-item">
          <div className="quality-item-label">Gaps</div>
          <div className={`quality-item-value ${quality.gap_count > 0 ? 'warn' : 'good'}`}>
            {quality.gap_count}
          </div>
        </div>
        <div className="quality-item">
          <div className="quality-item-label">Spikes</div>
          <div className={`quality-item-value ${quality.spike_count > 0 ? 'warn' : 'good'}`}>
            {quality.spike_count ?? 0}
          </div>
        </div>
        <div className="quality-item">
          <div className="quality-item-label">Channels</div>
          <div className="quality-item-value">
            {quality.channels?.length ?? quality.channels_found?.length ?? 0}
          </div>
        </div>
        {quality.low_variance_channels && quality.low_variance_channels.length > 0 && (
          <div className="quality-item">
            <div className="quality-item-label">Low Variance</div>
            <div className="quality-item-value warn">
              {quality.low_variance_channels.join(', ')}
            </div>
          </div>
        )}
      </div>
      {quality.date_range && (
        <div className="text-sm muted mt-8">Range: {quality.date_range}</div>
      )}
    </div>
  );
}
