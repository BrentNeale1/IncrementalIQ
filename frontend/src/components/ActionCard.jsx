import { getChannelName } from '../constants';

function getVariant(rec) {
  const t = (rec.type || '').toLowerCase();
  const msg = (rec.message || '').toLowerCase();
  if (t.includes('scale') || msg.includes('scale')) return 'scale';
  if (t.includes('experiment') || t.includes('test') || msg.includes('test')) return 'test';
  if (t.includes('reduce') || t.includes('underperform') || msg.includes('reduce')) return 'reduce';
  if (t.includes('geo') || msg.includes('geo')) return 'geo';
  return 'test';
}

function getTypeLabel(variant) {
  switch (variant) {
    case 'scale': return 'Scale';
    case 'test': return 'Test';
    case 'reduce': return 'Review';
    case 'geo': return 'Geo-test';
    default: return 'Action';
  }
}

export default function ActionCard({ recommendation }) {
  const variant = getVariant(recommendation);
  const channelName = recommendation.channel
    ? getChannelName(recommendation.channel)
    : null;

  return (
    <div className={`action-card ${variant}`}>
      <div className="action-card-type">{getTypeLabel(variant)}</div>
      <div className="action-card-title">
        {channelName || 'General'}
      </div>
      <div className="action-card-reason">{recommendation.message}</div>
    </div>
  );
}
