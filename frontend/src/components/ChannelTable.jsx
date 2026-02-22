import { getChannelColor, getChannelName } from '../constants';

function confidenceClass(conf) {
  if (!conf) return 'moderate';
  if (conf.startsWith('High')) return 'high';
  if (conf.startsWith('Low')) return 'low';
  return 'moderate';
}

export default function ChannelTable({ channels, showCI = false }) {
  if (!channels || channels.length === 0) return null;

  const maxPct = Math.max(...channels.map((c) => c.contribution_pct));

  return (
    <table className="channel-table">
      <thead>
        <tr>
          <th>Channel</th>
          <th>Share</th>
          {showCI && <th>CI Range (94%)</th>}
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {channels.map((ch) => (
          <tr key={ch.channel}>
            <td>
              <span
                className="channel-dot"
                style={{ background: getChannelColor(ch.channel) }}
              />
              <span className="channel-name">
                {ch.display_name || getChannelName(ch.channel)}
              </span>
            </td>
            <td>
              <div className="share-bar-container">
                <div className="share-bar">
                  <div
                    className="share-bar-fill"
                    style={{
                      width: `${maxPct > 0 ? (ch.contribution_pct / maxPct) * 100 : 0}%`,
                      background: getChannelColor(ch.channel),
                    }}
                  />
                </div>
                <span className="share-pct">
                  {ch.contribution_pct.toFixed(1)}%
                </span>
              </div>
            </td>
            {showCI && (
              <td>
                <span className="ci-range">
                  {ch.contribution_hdi_3 != null
                    ? `${ch.contribution_hdi_3.toFixed(1)}% – ${ch.contribution_hdi_97.toFixed(1)}%`
                    : '—'}
                </span>
              </td>
            )}
            <td>
              <span className={`confidence-pip ${confidenceClass(ch.confidence)}`}>
                {ch.confidence}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
