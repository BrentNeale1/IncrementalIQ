import Plot from 'react-plotly.js';
import { getChannelColor, getChannelName, PLOTLY_CONFIG, PLOTLY_LAYOUT_DEFAULTS } from '../../constants';

export default function ContributionBars({ channels, showErrorBars = false }) {
  if (!channels || channels.length === 0) return null;

  const sorted = [...channels].sort((a, b) => b.contribution_pct - a.contribution_pct);

  const trace = {
    type: 'bar',
    orientation: 'h',
    y: sorted.map((c) => c.display_name || getChannelName(c.channel)),
    x: sorted.map((c) => c.contribution_pct),
    marker: {
      color: sorted.map((c) => getChannelColor(c.channel)),
    },
    text: sorted.map((c) => `${c.contribution_pct.toFixed(1)}%`),
    textposition: 'outside',
    hovertemplate: '%{y}: %{x:.1f}%<extra></extra>',
  };

  if (showErrorBars) {
    trace.error_x = {
      type: 'data',
      symmetric: false,
      array: sorted.map((c) =>
        c.contribution_hdi_97 != null ? c.contribution_hdi_97 - c.contribution_pct : 0
      ),
      arrayminus: sorted.map((c) =>
        c.contribution_hdi_3 != null ? c.contribution_pct - c.contribution_hdi_3 : 0
      ),
      color: '#7a6f62',
      thickness: 1.5,
    };
  }

  return (
    <Plot
      data={[trace]}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: { text: 'Channel Contribution', font: { size: 14, family: 'Playfair Display' } },
        xaxis: { ...PLOTLY_LAYOUT_DEFAULTS.xaxis, title: '% of Revenue', ticksuffix: '%' },
        yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, automargin: true },
        margin: { l: 120, r: 60, t: 40, b: 40 },
        height: Math.max(240, sorted.length * 50),
      }}
      config={PLOTLY_CONFIG}
      style={{ width: '100%' }}
    />
  );
}
