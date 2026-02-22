import Plot from 'react-plotly.js';
import { getChannelColor, getChannelName, PLOTLY_CONFIG, PLOTLY_LAYOUT_DEFAULTS } from '../../constants';

export default function RevenueDecomposition({ channels, baselinePct }) {
  if (!channels || channels.length === 0) return null;

  // Build a simple stacked bar showing contribution breakdown
  const sorted = [...channels].sort((a, b) => b.contribution_pct - a.contribution_pct);

  const traces = sorted.map((ch) => ({
    type: 'bar',
    name: ch.display_name || getChannelName(ch.channel),
    x: ['Revenue'],
    y: [ch.contribution_pct],
    marker: { color: getChannelColor(ch.channel) },
    hovertemplate: `${ch.display_name || getChannelName(ch.channel)}: %{y:.1f}%<extra></extra>`,
  }));

  if (baselinePct > 0) {
    traces.push({
      type: 'bar',
      name: 'Baseline',
      x: ['Revenue'],
      y: [baselinePct],
      marker: { color: '#d4cec4' },
      hovertemplate: 'Baseline: %{y:.1f}%<extra></extra>',
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        barmode: 'stack',
        title: { text: 'Revenue Decomposition', font: { size: 14, family: 'Playfair Display' } },
        yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, title: '% of Revenue', ticksuffix: '%' },
        height: 300,
        showlegend: true,
        legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
        margin: { l: 50, r: 20, t: 40, b: 60 },
      }}
      config={PLOTLY_CONFIG}
      style={{ width: '100%' }}
    />
  );
}
