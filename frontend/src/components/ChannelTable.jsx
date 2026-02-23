import { getChannelColor, getChannelName, getControlColor, getControlName } from '../constants';

function confidenceClass(conf) {
  if (!conf) return 'moderate';
  if (conf.startsWith('High')) return 'high';
  if (conf.startsWith('Low')) return 'low';
  return 'moderate';
}

export default function ChannelTable({ channels, controls, showCI = false }) {
  if ((!channels || channels.length === 0) && (!controls || controls.length === 0)) return null;

  const allItems = [
    ...(channels || []).map((c) => ({ ...c, _type: 'channel' })),
  ];
  const controlItems = (controls || []).map((c) => ({ ...c, _type: 'control' }));

  const maxPct = Math.max(
    ...allItems.map((c) => c.contribution_pct),
    ...controlItems.map((c) => c.contribution_pct),
    0.1,
  );

  return (
    <table className="channel-table">
      <thead>
        <tr>
          <th>Source</th>
          <th>Share</th>
          {showCI && <th>CI Range (94%)</th>}
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {allItems.map((ch) => (
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
        {controlItems.length > 0 && (
          <tr>
            <td
              colSpan={showCI ? 4 : 3}
              style={{
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                fontWeight: 600,
                paddingTop: '12px',
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
              }}
            >
              Traffic Sources
            </td>
          </tr>
        )}
        {controlItems.map((ctrl) => (
          <tr key={ctrl.control}>
            <td>
              <span
                className="channel-dot"
                style={{ background: getControlColor(ctrl.control) }}
              />
              <span className="channel-name">
                {ctrl.display_name || getControlName(ctrl.control)}
              </span>
            </td>
            <td>
              <div className="share-bar-container">
                <div className="share-bar">
                  <div
                    className="share-bar-fill"
                    style={{
                      width: `${maxPct > 0 ? (ctrl.contribution_pct / maxPct) * 100 : 0}%`,
                      background: getControlColor(ctrl.control),
                    }}
                  />
                </div>
                <span className="share-pct">
                  {ctrl.contribution_pct.toFixed(1)}%
                </span>
              </div>
            </td>
            {showCI && (
              <td>
                <span className="ci-range">
                  {ctrl.contribution_hdi_3 != null
                    ? `${ctrl.contribution_hdi_3.toFixed(1)}% – ${ctrl.contribution_hdi_97.toFixed(1)}%`
                    : '—'}
                </span>
              </td>
            )}
            <td>
              <span className="confidence-pip moderate">—</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
