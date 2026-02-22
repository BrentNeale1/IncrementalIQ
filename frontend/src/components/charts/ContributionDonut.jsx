import Plot from 'react-plotly.js';
import { getChannelColor, getChannelName, PLOTLY_CONFIG, PLOTLY_LAYOUT_DEFAULTS } from '../../constants';

export default function ContributionDonut({ channels, baselinePct }) {
  if (!channels || channels.length === 0) return null;

  const labels = [
    ...channels.map((c) => c.display_name || getChannelName(c.channel)),
    'Baseline',
  ];
  const values = [
    ...channels.map((c) => c.contribution_pct),
    baselinePct ?? 0,
  ];
  const colors = [
    ...channels.map((c) => getChannelColor(c.channel)),
    '#d4cec4',
  ];

  return (
    <Plot
      data={[
        {
          type: 'pie',
          labels,
          values,
          hole: 0.55,
          marker: { colors },
          textinfo: 'label+percent',
          textposition: 'outside',
          textfont: { family: 'DM Sans', size: 11 },
          hovertemplate: '%{label}: %{value:.1f}%<extra></extra>',
          sort: false,
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: { text: 'Revenue Decomposition', font: { size: 14, family: 'Playfair Display' } },
        showlegend: false,
        height: 320,
        margin: { l: 20, r: 20, t: 40, b: 20 },
      }}
      config={PLOTLY_CONFIG}
      style={{ width: '100%' }}
    />
  );
}
