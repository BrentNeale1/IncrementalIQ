import { getTrustTier } from '../constants';

export default function TrustBadge({ tier }) {
  const info = getTrustTier(tier);
  const cls =
    tier === 'Model results are reliable'
      ? 'reliable'
      : tier === 'Use with caution'
        ? 'caution'
        : 'insufficient';

  return (
    <span className={`trust-badge ${cls}`}>{info.shortLabel}</span>
  );
}
